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

"""Tests for force_flip_invariance in the PyTorch TimesFM 2.5 model."""

import numpy as np
import torch

from timesfm import configs
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch


def _flip_quantile(x):
  # Keep the mean channel (index 0), reverse the nine quantile channels.
  return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)


class TestForceFlipInvariance:
  """force_flip_invariance must hold over the whole horizon, not just prefill."""

  def test_flip_invariance_holds_in_autoregressive_region(self):
    # With force_flip_invariance the model is symmetrized so that
    # forecast(-x) == -flip_quantile(forecast(x)) at every horizon step. The
    # property is independent of the weights, so a random init is enough to
    # check it. A missing quantile flip on the autoregressive branch breaks it
    # past the first output patch (128 steps).
    torch.manual_seed(0)
    tfm = TimesFM_2p5_200M_torch(torch_compile=False)
    tfm.model.eval()

    # max_horizon 256 > output patch length 128 so the AR branch is exercised.
    fc = configs.ForecastConfig(
        max_context=256,
        max_horizon=256,
        force_flip_invariance=True,
        use_continuous_quantile_head=False,
        infer_is_positive=False,  # skip the nonneg clamp so the identity is exact
        normalize_inputs=False,
        fix_quantile_crossing=False,
        return_backcast=False,
    )
    tfm.compile(fc)

    rng = np.random.RandomState(0)
    context = rng.standard_normal((1, 256)).astype(np.float32)  # mixed sign
    masks = np.zeros((1, 256), dtype=bool)

    with torch.no_grad():
      _, forecast_pos = tfm.compiled_decode(256, context, masks)
      _, forecast_neg = tfm.compiled_decode(256, -context, masks)

    forecast_pos = torch.from_numpy(np.asarray(forecast_pos))
    forecast_neg = torch.from_numpy(np.asarray(forecast_neg))
    expected = -_flip_quantile(forecast_pos)

    # The prefill region (first 128 steps) matched even before the fix; the AR
    # region (128:256) is where the missing flip showed up.
    ar_region = slice(128, 256)
    torch.testing.assert_close(
        forecast_neg[:, ar_region, :], expected[:, ar_region, :],
        atol=1e-4, rtol=1e-4,
    )
