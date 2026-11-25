# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ForecastConfig validation in TimesFM_2p5_200M_torch.compile."""

from types import SimpleNamespace

import pytest

from timesfm import ForecastConfig
from timesfm.timesfm_2p5 import timesfm_2p5_torch


@pytest.fixture
def model(monkeypatch):
  # Exercise compile() without building the 200M parameter module.
  monkeypatch.setattr(
      timesfm_2p5_torch, "TimesFM_2p5_200M_torch_module",
      lambda: SimpleNamespace(
          p=32, o=128, os=1024, q=10, device="cpu", device_count=1,
          config=SimpleNamespace(context_limit=16384),
      ),
  )
  return timesfm_2p5_torch.TimesFM_2p5_200M_torch(torch_compile=False)


@pytest.mark.parametrize("max_context", [0, -32])
def test_compile_rejects_non_positive_max_context(model, max_context):
  # max_context defaults to 0. `forecast` would then keep every series at its
  # full length (`value[-0:]`), so lengths that are not a multiple of the
  # patch size fail deep inside decode and no context cap is applied.
  with pytest.raises(ValueError, match="max_context"):
    model.compile(ForecastConfig(max_context=max_context, max_horizon=128))


def test_compile_accepts_positive_max_context(model):
  model.compile(ForecastConfig(max_context=100, max_horizon=128))
  assert model.forecast_config.max_context == 128
