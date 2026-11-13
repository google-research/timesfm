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

"""Future-patch extraction preserves ordering, masks and gradients."""

import pytest
import torch

from timesfm3.torch.util import get_output_patch_via_roll


@pytest.mark.parametrize("rolls", [0, 1, 2, 7])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bool])
def test_future_patch_values_and_wrap_mask(rolls, noncontiguous, dtype):
  values = torch.arange(2 * 3 * 5 * 8).reshape(2, 3, 5, 8)
  if dtype == torch.bool:
    values = values % 3 == 0
  else:
    values = values.to(dtype)
  if noncontiguous:
    values = values[..., ::2]
  output, mask = get_output_patch_via_roll(values, rolls)
  expected = (
    torch.cat(
      [torch.roll(values, -offset, 2) for offset in range(1, rolls + 1)], dim=-1
    )
    if rolls
    else values[..., :0]
  )
  torch.testing.assert_close(output, expected, rtol=0, atol=0)
  for patch in range(5):
    valid_length = min(rolls, 4 - patch) * values.shape[-1]
    assert not mask[0, 0, patch, :valid_length].any()
    assert mask[0, 0, patch, valid_length:].all()


def test_future_patch_gradients():
  torch.manual_seed(0)
  values = torch.randn(2, 3, 5, 4, dtype=torch.float64, requires_grad=True)
  result, _ = get_output_patch_via_roll(values, 7)
  expected = torch.cat(
    [torch.roll(values, -offset, 2) for offset in range(1, 8)], dim=-1
  )
  weight = torch.randn_like(result)
  actual_grad = torch.autograd.grad((result * weight).sum(), values, retain_graph=True)[
    0
  ]
  expected_grad = torch.autograd.grad((expected * weight).sum(), values)[0]
  torch.testing.assert_close(actual_grad, expected_grad)
