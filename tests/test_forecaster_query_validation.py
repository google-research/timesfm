# Copyright 2026 Google LLC
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

"""Reject misaligned forecast inputs before silently truncating or decoding."""

from unittest import mock

import numpy as np
import pytest
import torch

from timesfm3.torch import timesfm3_forecaster as api
from timesfm3.torch.timesfm3_forecaster_test import _RecordingFakeModel


@pytest.fixture
def forecaster():
  with mock.patch.object(api.TimesFM3Forecaster, "_init_model"):
    result = api.TimesFM3Forecaster(
      api.ModelConfig(
        per_core_batch_size=1,
        input_patch_length=8,
        output_patch_length=8,
        median_quantile_index=1,
      )
    )
  result.model = _RecordingFakeModel()
  result.device = torch.device("cpu")
  return result


@pytest.mark.parametrize(
  "name", ["ts_ids", "past_only_covariates", "past_future_covariates"]
)
@pytest.mark.parametrize("length", [0, 1, 3])
def test_batch_metadata_lengths(forecaster, name, length):
  values = ["series"] * length if name == "ts_ids" else [None] * length
  with pytest.raises(ValueError, match=name):
    list(forecaster.predict_batch([np.arange(8)] * 2, 4, **{name: values}))
  assert not forecaster.model.calls


@pytest.mark.parametrize(
  "name,width",
  [
    ("past_only_covariates", 7),
    ("past_only_covariates", 9),
    ("past_future_covariates", 11),
    ("past_future_covariates", 13),
  ],
)
def test_covariates_align_with_context_and_horizon(forecaster, name, width):
  with pytest.raises(ValueError, match=name):
    list(forecaster.predict_batch([np.arange(8)], 4, **{name: [np.arange(width)]}))
  assert not forecaster.model.calls


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon(forecaster, horizon):
  with pytest.raises(ValueError, match="horizon"):
    list(forecaster.predict_batch([np.arange(8)], horizon))
  assert not forecaster.model.calls


@pytest.mark.parametrize(
  "context",
  [
    np.array(1.0),
    np.ones((1, 1, 8)),
    np.ones((0, 8)),
    np.array([]),
    np.empty((1, 0)),
  ],
)
def test_invalid_context_shape(forecaster, context):
  with pytest.raises(ValueError, match="contexts"):
    list(forecaster.predict_batch([context], 4))
  assert not forecaster.model.calls


def test_validation_precedes_leading_nan_trimming(forecaster):
  context = np.array([np.nan, np.nan, 1, 2, 3, 4, 5, 6], np.float32)
  out = forecaster.predict(
    context,
    4,
    past_only_covariates=np.arange(8),
    past_future_covariates=np.arange(12),
    padding_mode="edge",
  )
  assert out.forecast.shape == (4,)
  np.testing.assert_array_equal(
    forecaster.model.calls[0]["past_only_covariates"][0, 0, -6:], np.arange(2, 8)
  )
