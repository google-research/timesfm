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

"""Each forecast preserves the dimensionality of its own input series."""

from unittest import mock

import numpy as np
import pytest
import torch

from timesfm3.torch import timesfm3_forecaster as api
from timesfm3.torch.evaluator import TimesFM3Evaluator
from timesfm3.torch.timesfm3_forecaster_test import FakeModel


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("cls", [api.TimesFM3Forecaster, TimesFM3Evaluator])
def test_mixed_univariate_representations(reverse, positive, symmetric, cls):
  with mock.patch.object(api.TimesFM3Forecaster, "_init_model"):
    forecaster = cls(
      api.ModelConfig(
        per_core_batch_size=2,
        input_patch_length=8,
        output_patch_length=8,
        median_quantile_index=1,
      )
    )
  forecaster.model = FakeModel()
  forecaster.device = torch.device("cpu")
  contexts = [np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32)[None]]
  if reverse:
    contexts.reverse()
  outputs = list(
    forecaster.predict_batch(
      contexts,
      horizon=4,
      return_quantiles=True,
      make_positive=positive,
      use_symmetric_averaging=symmetric,
    )
  )
  for context, output in zip(contexts, outputs):
    expected_shape = (4,) if context.ndim == 1 else (1, 4)
    assert output.forecast.shape == expected_shape
    assert output.quantiles.shape == expected_shape + (3,)
    np.testing.assert_array_equal(output.forecast, 0.0 if symmetric else 2.0)


class _NegativeFakeModel(FakeModel):
  """Returns -1 everywhere so non-negativity clamping is observable."""

  def decode(self, target, *args, horizon=8, **kwargs):
    b, v, _ = target.shape
    return -torch.ones((b, v, horizon, self.quantiles), dtype=torch.float32)


@pytest.mark.parametrize(
  "context, expected_shape",
  [
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (4,)),
    ([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], (2, 4)),
  ],
)
@pytest.mark.parametrize("cls", [api.TimesFM3Forecaster, TimesFM3Evaluator])
def test_make_positive_accepts_list_contexts(context, expected_shape, cls):
  # make_positive=True is the evaluator default, so a list context must work.
  with mock.patch.object(api.TimesFM3Forecaster, "_init_model"):
    forecaster = cls(
      api.ModelConfig(
        per_core_batch_size=2,
        input_patch_length=8,
        output_patch_length=8,
        median_quantile_index=1,
      )
    )
  forecaster.model = _NegativeFakeModel()
  forecaster.device = torch.device("cpu")
  (output,) = forecaster.predict_batch(
    [context],
    horizon=4,
    make_positive=True,
    use_symmetric_averaging=False,
  )
  assert output.forecast.shape == expected_shape
  np.testing.assert_array_equal(output.forecast, 0.0)
