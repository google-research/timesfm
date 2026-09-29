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

"""Stitched output values and gradients match the existing patch layout."""

import pytest
import torch

from timesfm3.torch.util import stitch_patches


def reference_stitch(predictions, patch_len):
  count = predictions.shape[2]
  overlap = predictions.shape[3] - patch_len
  if count == 1:
    return predictions[:, :, 0]
  weight = torch.linspace(
    1.0, 0.0, overlap, device=predictions.device, dtype=predictions.dtype
  )[None, None, None, :, None]
  pieces = [predictions[:, :, 0, :patch_len]]
  for index in range(1, count):
    previous = predictions[:, :, index - 1, patch_len:]
    following = predictions[:, :, index, :overlap]
    pieces.append(weight[:, :, 0] * previous + (1.0 - weight[:, :, 0]) * following)
    pieces.append(predictions[:, :, index, overlap:patch_len])
  pieces.append(predictions[:, :, -1, patch_len:])
  return torch.cat(pieces, dim=2)


@pytest.mark.parametrize("count", [1, 2, 8])
@pytest.mark.parametrize("overlap", [0, 1, 4])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_stitch_output_values_and_gradients(count, overlap, noncontiguous):
  torch.manual_seed(0)
  source = torch.randn(2, 3, count, 8 + overlap, 4, dtype=torch.float64)
  if noncontiguous:
    source = source.transpose(0, 1)
  predictions = source.detach().requires_grad_()
  actual = stitch_patches(predictions, 8)
  expected = reference_stitch(predictions, 8)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  weights = torch.randn_like(actual)
  actual_grad = torch.autograd.grad(
    (actual * weights).sum(), predictions, retain_graph=True
  )[0]
  expected_grad = torch.autograd.grad((expected * weights).sum(), predictions)[0]
  torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
