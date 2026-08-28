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

"""Tests for PyTorch cpm_iterative_revin_refine."""

import unittest

import numpy as np
import torch

from . import cpm_revin_refine as torch_lib


class CpmRevinRefineTest(unittest.TestCase):
  def test_shape(self):
    b, v, n = 2, 3, 8
    rolls, patch_len, num_q = 2, 2, 3
    oq = rolls * patch_len * num_q
    raw_logits = torch.zeros((b, v, n, oq))
    revin_n = torch.ones((b, v, n))
    revin_mu = torch.zeros((b, v, n))
    revin_sigma = torch.ones((b, v, n))
    patch_cpm_mask = torch.zeros((b, n), dtype=torch.bool)

    refined_mu, refined_sigma = torch_lib.cpm_iterative_revin_refine(
      raw_logits=raw_logits,
      revin_n=revin_n,
      revin_mu=revin_mu,
      revin_sigma=revin_sigma,
      patch_cpm_mask=patch_cpm_mask,
      median_q_idx=num_q // 2,
      rolls=rolls,
      patch_len=patch_len,
      num_quantiles=num_q,
    )
    self.assertEqual(refined_mu.shape, (b, v, n))
    self.assertEqual(refined_sigma.shape, (b, v, n))

  def test_no_cpm_mask_identity(self):
    b, v, n = 1, 2, 6
    rolls, patch_len, num_q = 2, 4, 3
    oq = rolls * patch_len * num_q

    raw_logits = torch.randn(b, v, n, oq)
    revin_n = torch.full((b, v, n), 4.0)
    revin_mu = torch.randn(b, v, n)
    revin_sigma = torch.abs(torch.randn(b, v, n)) + 0.1
    patch_cpm_mask = torch.zeros((b, n), dtype=torch.bool)

    refined_mu, refined_sigma = torch_lib.cpm_iterative_revin_refine(
      raw_logits=raw_logits,
      revin_n=revin_n,
      revin_mu=revin_mu,
      revin_sigma=revin_sigma,
      patch_cpm_mask=patch_cpm_mask,
      median_q_idx=num_q // 2,
      rolls=rolls,
      patch_len=patch_len,
      num_quantiles=num_q,
    )
    np.testing.assert_allclose(refined_mu.numpy(), revin_mu.numpy())
    np.testing.assert_allclose(refined_sigma.numpy(), revin_sigma.numpy())

  def test_cpm_mask_modifies_cpm_positions_only(self):
    for b, v, n, rolls, patch_len, num_q in [
      (1, 1, 8, 2, 4, 3),
      (2, 3, 16, 2, 32, 9),
      (2, 2, 12, 4, 8, 5),
    ]:
      torch.manual_seed(42)
      oq = rolls * patch_len * num_q

      raw_logits = torch.randn(b, v, n, oq)
      revin_n = torch.full((b, v, n), 10.0)
      revin_mu = torch.randn(b, v, n)
      revin_sigma = torch.abs(torch.randn(b, v, n)) + 0.5

      # First 4 patches non-CPM, remaining CPM
      patch_cpm_mask = torch.zeros((b, n), dtype=torch.bool)
      patch_cpm_mask[:, 4:] = True

      median_q_idx = num_q // 2

      refined_mu, refined_sigma = torch_lib.cpm_iterative_revin_refine(
        raw_logits=raw_logits,
        revin_n=revin_n,
        revin_mu=revin_mu,
        revin_sigma=revin_sigma,
        patch_cpm_mask=patch_cpm_mask,
        median_q_idx=median_q_idx,
        rolls=rolls,
        patch_len=patch_len,
        num_quantiles=num_q,
      )

      # Non-CPM positions should be identical to inputs
      np.testing.assert_allclose(
        refined_mu[:, :, :4].numpy(), revin_mu[:, :, :4].numpy()
      )
      np.testing.assert_allclose(
        refined_sigma[:, :, :4].numpy(), revin_sigma[:, :, :4].numpy()
      )

      # Refined outputs should be finite
      self.assertTrue(torch.isfinite(refined_mu).all().item())
      self.assertTrue(torch.isfinite(refined_sigma).all().item())
      self.assertTrue((refined_sigma > 0).all().item())


if __name__ == "__main__":
  unittest.main()
