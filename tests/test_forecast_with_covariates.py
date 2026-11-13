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

"""Tests for forecast_with_covariates in the TimesFM 2.5 model."""

import numpy as np
import pytest
import torch

from timesfm import configs
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

# The XReg module needs the optional `timesfm[xreg]` dependencies.
pytest.importorskip("jax")
pytest.importorskip("sklearn")

_MAX_CONTEXT = 128
_HORIZON = 16


@pytest.fixture(scope="module")
def tfm():
  # The shapes under test do not depend on the weights, so a random init is
  # enough.
  torch.manual_seed(0)
  model = TimesFM_2p5_200M_torch(torch_compile=False)
  model.model.eval()
  model.compile(
      configs.ForecastConfig(
          max_context=_MAX_CONTEXT,
          max_horizon=128,
          return_backcast=True,
      )
  )
  return model


def _forecast(tfm, context_lens, normalize=True):
  rng = np.random.default_rng(0)
  inputs = [rng.normal(size=n) + 10.0 for n in context_lens]
  covariates = [rng.normal(size=n + _HORIZON) for n in context_lens]
  with torch.no_grad():
    point, quantile = tfm.forecast_with_covariates(
        inputs,
        dynamic_numerical_covariates={"x": covariates},
        xreg_mode="timesfm + xreg",
        normalize_xreg_target_per_input=normalize,
    )
  return np.asarray(point), np.asarray(quantile)


class TestTimesFMPlusXReg:
  """The "timesfm + xreg" mode must handle contexts of any length."""

  @pytest.mark.parametrize(
      "context_lens, normalize",
      [
          # Longer than max_context: the backcast covers only the last 128.
          ([200], True),
          ([200], False),
          ([_MAX_CONTEXT], True),
          # The second input is shorter than one input patch, so it has no
          # backcast to fit on.
          ([100, 20], True),
          ([100, 20], False),
      ],
  )
  def test_context_length(self, tfm, context_lens, normalize):
    point, quantile = _forecast(tfm, context_lens, normalize)

    assert point.shape == (len(context_lens), _HORIZON)
    assert quantile.shape[:2] == (len(context_lens), _HORIZON)
    assert np.isfinite(point).all()
    assert np.isfinite(quantile).all()

  def test_all_contexts_within_one_patch(self, tfm):
    with pytest.raises(ValueError, match="input patch length"):
      _forecast(tfm, [20])


@pytest.mark.parametrize("xreg_mode", ["xreg + timesfm", "timesfm + xreg"])
def test_nans_in_inputs_are_handled_like_forecast(tfm, xreg_mode):
  rng = np.random.default_rng(0)
  context_lens = [100, 80]
  inputs = [rng.normal(size=n) + 10.0 for n in context_lens]
  covariates = [rng.normal(size=n + _HORIZON) for n in context_lens]
  # Leading NaNs are dropped and inner NaNs are linearly interpolated.
  with_nans = inputs[0].copy()
  with_nans[:10] = np.nan
  with_nans[60] = np.nan
  cleaned = inputs[0][10:].copy()
  cleaned[50] = (cleaned[49] + cleaned[51]) / 2

  def forecast(first_input, first_covariate):
    with torch.no_grad():
      return tfm.forecast_with_covariates(
          [first_input, inputs[1]],
          dynamic_numerical_covariates={"x": [first_covariate, covariates[1]]},
          xreg_mode=xreg_mode,
      )

  point, quantile = forecast(with_nans, covariates[0])
  expected_point, expected_quantile = forecast(cleaned, covariates[0][10:])

  np.testing.assert_allclose(point, expected_point, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(quantile, expected_quantile, rtol=1e-5, atol=1e-5)
