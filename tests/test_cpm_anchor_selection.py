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

"""CPM refinement uses each example's own anchor offset and reset points."""

import numpy as np
import pytest
import torch

from timesfm3.torch.cpm_revin_refine import cpm_iterative_revin_refine


def numpy_reference(logits, counts, means, sigmas, mask, rolls, patch_len):
  batch, variates, patches, _ = logits.shape
  medians = logits.reshape(batch, variates, patches, rolls, patch_len, 3)[..., 1]
  out_mean, out_sigma = means.copy(), sigmas.copy()
  for b in range(batch):
    for v in range(variates):
      n, mean, sigma, offset = 0.0, 0.0, 0.0, 0
      anchor = np.zeros((rolls, patch_len))
      for i in range(patches):
        if mask[b, i]:
          values = anchor[offset]
          new_n = n + patch_len
          new_mean = (n * mean + values.sum()) / new_n
          variance = (
            n * (sigma**2 + (mean - new_mean) ** 2) + ((values - new_mean) ** 2).sum()
          ) / new_n
          n, mean, sigma = new_n, new_mean, np.sqrt(variance)
          offset = (offset + 1) % rolls
        else:
          n, mean, sigma = counts[b, v, i], means[b, v, i], sigmas[b, v, i]
          offset = 0
        out_mean[b, v, i], out_sigma[b, v, i] = mean, sigma
        if offset == 0:
          anchor = np.clip(medians[b, v, i] * sigma + mean, -1e9, 1e9)
  return out_mean, out_sigma


@pytest.mark.parametrize("rolls", [1, 2, 4, 8])
def test_heterogeneous_anchor_offsets_against_numpy(rolls):
  rng = np.random.default_rng(42)
  batch, variates, patches, patch_len = 3, 2, 12, 4
  logits = rng.normal(size=(batch, variates, patches, rolls * patch_len * 3)).astype(
    np.float32
  )
  counts = np.full((batch, variates, patches), 10, np.float32)
  means = rng.normal(size=counts.shape).astype(np.float32)
  sigmas = np.abs(rng.normal(size=counts.shape)).astype(np.float32) + 0.5
  mask = np.array(
    [
      [False] * 3 + [True] * 9,
      [False, True, True, False, True, True, True, False, True, True, True, True],
      [False] * 12,
    ]
  )
  actual = cpm_iterative_revin_refine(
    *map(torch.from_numpy, [logits, counts, means, sigmas, mask]),
    median_q_idx=1,
    rolls=rolls,
    patch_len=patch_len,
    num_quantiles=3,
  )
  expected = numpy_reference(logits, counts, means, sigmas, mask, rolls, patch_len)
  for result, reference in zip(actual, expected):
    np.testing.assert_allclose(result.numpy(), reference, rtol=1e-5, atol=1e-6)
