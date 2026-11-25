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

"""Tests for ForecastConfig validation in TimesFM 2.5 Flax compile."""

from types import SimpleNamespace

import pytest

from timesfm import ForecastConfig


@pytest.fixture
def model(monkeypatch):
  flax_backend = pytest.importorskip(
    "timesfm.timesfm_2p5.timesfm_2p5_flax",
    reason="TimesFM Flax extras are not installed",
    exc_type=ImportError,
  )
  monkeypatch.setattr(
    flax_backend,
    "TimesFM_2p5_200M_flax_module",
    lambda: SimpleNamespace(
      p=32,
      o=128,
      os=1024,
      num_devices=1,
      config=SimpleNamespace(context_limit=16384),
      compile=lambda **kwargs: None,
    ),
  )
  return flax_backend.TimesFM_2p5_200M_flax()


@pytest.mark.parametrize("max_context", [0, -32])
def test_flax_compile_rejects_non_positive_max_context(model, max_context):
  with pytest.raises(ValueError, match="max_context"):
    model.compile(ForecastConfig(max_context=max_context, max_horizon=128), dryrun=False)


def test_flax_compile_accepts_positive_max_context(model):
  model.compile(ForecastConfig(max_context=100, max_horizon=128), dryrun=False)
  assert model.forecast_config.max_context == 128
