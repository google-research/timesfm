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

"""Running patch statistics agree with a direct weighted reference."""

import numpy as np
import pytest
import torch

from timesfm3.torch.util import update_running_stats


@pytest.mark.parametrize("initial_count", [0.0, 4.0])
@pytest.mark.parametrize("mask_pattern", ["none", "partial", "all"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_running_stats_matches_direct_reference(initial_count, mask_pattern, dtype):
  torch.manual_seed(0)
  data = torch.randn(2, 3, 16, dtype=dtype)
  mask = torch.zeros_like(data, dtype=torch.bool)
  if mask_pattern == "partial":
    mask[..., ::3] = True
  elif mask_pattern == "all":
    mask[:] = True
  data = data.masked_fill(mask, float("nan")).requires_grad_()
  count = torch.full((2, 3), initial_count)
  mean = torch.full((2, 3), 1.25 if initial_count else 0.0)
  std = torch.full((2, 3), 0.75 if initial_count else 0.0)

  new_count, new_mean, new_std = update_running_stats(
    count, mean, std, data, mask
  )
  values = data.detach().numpy()
  masks = mask.numpy()
  for b in range(2):
    for v in range(3):
      valid = values[b, v][~masks[b, v]]
      total_count = initial_count + len(valid)
      expected_mean = (
        (initial_count * mean[b, v].item() + valid.sum()) / total_count
        if total_count else 0.0
      )
      expected_var = (
        (
          initial_count * (
            std[b, v].item() ** 2 + (mean[b, v].item() - expected_mean) ** 2
          )
          + np.square(valid - expected_mean).sum()
        ) / total_count
        if total_count else 0.0
      )
      assert new_count[b, v].item() == total_count
      np.testing.assert_allclose(new_mean[b, v].item(), expected_mean, rtol=1e-5)
      np.testing.assert_allclose(
        new_std[b, v].item(), np.sqrt(expected_var), rtol=1e-5, atol=1e-6
      )

  if mask_pattern != "all":
    (new_mean.sum() + new_std.sum()).backward()
    assert torch.isfinite(data.grad[~mask]).all()
    assert data.grad[~mask].abs().sum() > 0
    assert data.grad[mask].abs().sum() == 0
