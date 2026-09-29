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

"""Mixed covariate batches must keep the requested forecast horizon."""

from unittest import mock

import numpy as np
import pytest
import torch

from timesfm3.torch import timesfm3_forecaster as api
from timesfm3.torch.timesfm3_forecaster_test import _RecordingFakeModel


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("padding_mode", ["none", "edge"])
def test_optional_covariates_across_batches(batch_size, symmetric, padding_mode):
  config = api.ModelConfig(
    per_core_batch_size=batch_size,
    input_patch_length=8,
    output_patch_length=8,
    median_quantile_index=1,
  )
  with mock.patch.object(api.TimesFM3Forecaster, "_init_model"):
    forecaster = api.TimesFM3Forecaster(config)
  forecaster.model = _RecordingFakeModel()
  forecaster.device = torch.device("cpu")
  contexts = [np.arange(12, dtype=np.float32) + i for i in range(3)]
  covariates = [None, np.arange(17, dtype=np.float32), None]
  outputs = list(
    forecaster.predict_batch(
      contexts,
      horizon=5,
      past_future_covariates=covariates,
      ts_ids=["a", "b", "c"],
      return_quantiles=True,
      use_symmetric_averaging=symmetric,
      padding_mode=padding_mode,
    )
  )

  assert [out.ts_id for out in outputs] == ["a", "b", "c"]
  for out in outputs:
    assert out.forecast.shape == (5,)
    assert out.quantiles.shape == (5, 3)
    np.testing.assert_array_equal(out.forecast, 0.0 if symmetric else 2.0)
