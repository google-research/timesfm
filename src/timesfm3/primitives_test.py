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

"""Tests for TimesFM3 PyTorch primitives."""

import math
import unittest

import numpy as np
import torch

from . import configs
from . import dense as torch_dense
from . import normalization as torch_norm
from . import transformations as torch_trans
from . import util as torch_util


class TransformationsTest(unittest.TestCase):
  """Tests reversible transformations."""

  def test_signed_log(self):
    x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=torch.float32)
    y = torch_trans.signed_log(x)
    expected_y = torch.sign(x) * torch.log1p(torch.abs(x))
    np.testing.assert_allclose(y.numpy(), expected_y.numpy(), atol=1e-6)

    # Reverse round-trip
    z = torch_trans.signed_log(y, reverse=True)
    np.testing.assert_allclose(z.numpy(), x.numpy(), atol=1e-5)

  def test_signed_sqrt(self):
    x = torch.tensor([-9.0, -1.0, 0.0, 1.0, 16.0], dtype=torch.float32)
    y = torch_trans.signed_sqrt(x)
    expected_y = torch.sign(x) * torch.sqrt(torch.abs(x))
    np.testing.assert_allclose(y.numpy(), expected_y.numpy(), atol=1e-6)

    # Reverse round-trip
    z = torch_trans.signed_sqrt(y, reverse=True)
    np.testing.assert_allclose(z.numpy(), x.numpy(), atol=1e-5)

  def test_identity(self):
    x = torch.randn(2, 3, 4)
    y = torch_trans.identity(x)
    np.testing.assert_allclose(y.numpy(), x.numpy())
    z = torch_trans.identity(y, reverse=True)
    np.testing.assert_allclose(z.numpy(), x.numpy())

  def test_max_output(self):
    value_clip = 10.0
    self.assertAlmostEqual(
      torch_trans.max_output("signed_log", value_clip).item(),
      math.log1p(value_clip),
      places=5,
    )
    self.assertAlmostEqual(
      torch_trans.max_output("signed_sqrt", value_clip).item(),
      math.sqrt(value_clip),
      places=5,
    )
    self.assertAlmostEqual(
      torch_trans.max_output("identity", value_clip).item(),
      value_clip,
      places=5,
    )


class NormalizationTest(unittest.TestCase):
  """Tests PerDimScale."""

  def test_per_dim_scale(self):
    num_dims = 8
    mod = torch_norm.PerDimScale(num_dims=num_dims)
    x = torch.randn(2, 3, num_dims)
    y = mod(x)
    self.assertEqual(y.shape, x.shape)
    # PerDimScale scales by RECIPROCAL_OF_SOFTPLUS_0 / sqrt(num_dims) * softplus(0)
    # Since softplus(0) = ln(2) = 0.693147, and RECIPROCAL_OF_SOFTPLUS_0 = 1 / ln(2),
    # at init it approximately scales by 1 / sqrt(num_dims).
    expected_scale = 1.0 / math.sqrt(num_dims)
    np.testing.assert_allclose(
      y.detach().numpy(), (x * expected_scale).numpy(), atol=1e-4
    )


class ResidualBlockTest(unittest.TestCase):
  """Tests ResidualBlock."""

  def test_residual_block_identity_skip(self):
    input_dim = 8
    hidden_dims = 16
    config = configs.ResidualBlockConfig(
      hidden_dims=hidden_dims,
      output_dims=input_dim,
      use_bias=True,
      activation="relu",
      identity_skip=True,
      prenorm="none",
    )
    mod = torch_dense.ResidualBlock(config=config)
    x = torch.randn(2, input_dim)
    out = mod(x)
    self.assertEqual(out.shape, (2, input_dim))

  def test_residual_block_projection_skip(self):
    input_dim = 8
    hidden_dims = 16
    output_dims = 12
    config = configs.ResidualBlockConfig(
      hidden_dims=hidden_dims,
      output_dims=output_dims,
      use_bias=True,
      activation="relu",
      identity_skip=False,
      prenorm="none",
    )
    mod = torch_dense.ResidualBlock(config=config)
    x = torch.randn(2, input_dim)
    out = mod(x)
    self.assertEqual(out.shape, (2, output_dims))

  def test_residual_block_with_prenorm(self):
    input_dim = 8
    hidden_dims = 16
    config = configs.ResidualBlockConfig(
      hidden_dims=hidden_dims,
      output_dims=input_dim,
      use_bias=False,
      activation="relu",
      identity_skip=True,
      prenorm="rms",
    )
    mod = torch_dense.ResidualBlock(config=config)
    x = torch.randn(2, input_dim)
    out = mod(x)
    self.assertEqual(out.shape, (2, input_dim))


class UtilTest(unittest.TestCase):
  """Tests utility functions."""

  def test_activation_fns(self):
    x = torch.tensor([-1.0, 0.0, 1.0])

    relu = torch_util.get_activation_fn("relu")
    np.testing.assert_allclose(relu(x).numpy(), [0.0, 0.0, 1.0])

    silu = torch_util.get_activation_fn("silu")
    expected_silu = x * torch.sigmoid(x)
    np.testing.assert_allclose(silu(x).numpy(), expected_silu.numpy())

    swish = torch_util.get_activation_fn("swish")
    np.testing.assert_allclose(swish(x).numpy(), expected_silu.numpy())

    swiglu = torch_util.get_activation_fn("swiglu")
    np.testing.assert_allclose(swiglu(x).numpy(), expected_silu.numpy())

  def test_decode_cache(self):
    num_layers = 2
    batch_size = 2
    num_variates = 3
    num_total_input_patches = 10
    num_heads = 4
    head_dim = 8

    caches = torch_util.DecodeCache.init_decode_cache(
      num_layers=num_layers,
      batch_size=batch_size,
      num_variates=num_variates,
      num_total_input_patches=num_total_input_patches,
      num_heads=num_heads,
      head_dim=head_dim,
    )

    self.assertEqual(len(caches), num_layers)
    leading_size = batch_size * num_variates

    for cache in caches:
      self.assertEqual(cache.next_index.shape, (leading_size,))
      self.assertEqual(cache.num_front_masked.shape, (leading_size,))
      self.assertEqual(
        cache.key.shape,
        (leading_size, num_total_input_patches, num_heads, head_dim),
      )
      self.assertEqual(
        cache.value.shape,
        (leading_size, num_total_input_patches, num_heads, head_dim),
      )
      np.testing.assert_allclose(cache.next_index.numpy(), 0)
      np.testing.assert_allclose(cache.num_front_masked.numpy(), 0)
      np.testing.assert_allclose(cache.key.numpy(), 0.0)
      np.testing.assert_allclose(cache.value.numpy(), 0.0)

  def test_update_running_stats_single_batch_no_mask(self):
    n, mu, sigma = 0.0, 0.0, 0.0
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    mask = torch.zeros_like(x, dtype=torch.bool)
    new_n, new_mu, new_sigma = torch_util.update_running_stats(
      torch.tensor([[n]]),
      torch.tensor([[mu]]),
      torch.tensor([[sigma]]),
      x,
      mask,
    )
    self.assertAlmostEqual(new_n[0, 0].item(), 3.0)
    self.assertAlmostEqual(new_mu[0, 0].item(), 2.0)
    self.assertAlmostEqual(
      new_sigma[0, 0].item(), torch.std(x, unbiased=False).item(), places=6
    )

  def test_update_running_stats_multiple_batches_no_mask(self):
    x1 = torch.tensor([[[1.0, 2.0, 3.0]]])
    mask1 = torch.zeros_like(x1, dtype=torch.bool)
    x2 = torch.tensor([[[4.0, 5.0, 6.0]]])
    mask2 = torch.zeros_like(x2, dtype=torch.bool)

    n1, mu1, sigma1 = torch_util.update_running_stats(
      torch.tensor([[0.0]]),
      torch.tensor([[0.0]]),
      torch.tensor([[0.0]]),
      x1,
      mask1,
    )
    n2, mu2, sigma2 = torch_util.update_running_stats(n1, mu1, sigma1, x2, mask2)

    self.assertAlmostEqual(n2[0, 0].item(), 6.0)
    self.assertAlmostEqual(mu2[0, 0].item(), 3.5)
    concat_x = torch.cat([x1, x2], dim=-1)
    self.assertAlmostEqual(
      sigma2[0, 0].item(),
      torch.std(concat_x, unbiased=False).item(),
      places=6,
    )

  def test_update_running_stats_with_mask(self):
    n, mu, sigma = 0.0, 0.0, 0.0
    x = torch.tensor([[[1.0, 100.0, 2.0, 3.0]]])
    mask = torch.tensor([[[False, True, False, False]]])
    new_n, new_mu, new_sigma = torch_util.update_running_stats(
      torch.tensor([[n]]),
      torch.tensor([[mu]]),
      torch.tensor([[sigma]]),
      x,
      mask,
    )
    self.assertAlmostEqual(new_n[0, 0].item(), 3.0)
    self.assertAlmostEqual(new_mu[0, 0].item(), 2.0)
    valid_x = torch.tensor([1.0, 2.0, 3.0])
    self.assertAlmostEqual(
      new_sigma[0, 0].item(),
      torch.std(valid_x, unbiased=False).item(),
      places=6,
    )

  def test_revin_no_reverse(self):
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    mu = torch.tensor([[2.0]])
    sigma = torch.tensor([[1.0]])
    normalized_x = torch_util.revin(x, mu, sigma, reverse=False)
    expected = torch.tensor([[[-1.0, 0.0, 1.0]]])
    np.testing.assert_allclose(normalized_x.numpy(), expected.numpy(), atol=1e-6)

  def test_revin_reverse(self):
    x_normalized = torch.tensor([[[-1.0, 0.0, 1.0]]])
    mu = torch.tensor([[2.0]])
    sigma = torch.tensor([[1.0]])
    denormalized_x = torch_util.revin(x_normalized, mu, sigma, reverse=True)
    expected = torch.tensor([[[1.0, 2.0, 3.0]]])
    np.testing.assert_allclose(denormalized_x.numpy(), expected.numpy(), atol=1e-6)

  def test_revin_near_zero_sigma(self):
    x = torch.tensor([[[2.0, 2.0, 2.0]]])
    mu = torch.tensor([[2.0]])
    sigma = torch.tensor([[1e-7]])
    normalized_x = torch_util.revin(x, mu, sigma, reverse=False)
    expected = torch.tensor([[[0.0, 0.0, 0.0]]])
    np.testing.assert_allclose(normalized_x.numpy(), expected.numpy(), atol=1e-6)

  def test_get_output_patch_via_roll(self):
    x = torch.tensor([[[[1, 2], [3, 4], [5, 6], [7, 8]]]], dtype=torch.float32)
    rolls = 2
    expected_output = torch.tensor(
      [[[[3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 1, 2], [1, 2, 3, 4]]]],
      dtype=torch.float32,
    )
    output, _ = torch_util.get_output_patch_via_roll(x, rolls)
    np.testing.assert_allclose(output.numpy(), expected_output.numpy(), atol=1e-6)

  def test_get_running_stats_single_segment(self):
    values = torch.tensor([[[[1, 3], [5, 7]]]], dtype=torch.float32)
    masks = torch.zeros_like(values, dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0]], dtype=torch.int32)
    _, mu, sigma = torch_util.get_running_stats(values, masks, segment_ids=segment_ids)

    expected_mu = torch.tensor([[[2.0, 4.0]]])
    expected_sigma = torch.tensor([[[1.0, 2.236068]]])
    np.testing.assert_allclose(mu.numpy(), expected_mu.numpy(), atol=1e-6)
    np.testing.assert_allclose(sigma.numpy(), expected_sigma.numpy(), atol=1e-6)

  def test_get_running_stats_multiple_segments(self):
    values = torch.tensor([[[[1, 3], [10, 10], [5, 7], [4, 6]]]], dtype=torch.float32)
    masks = torch.zeros_like(values, dtype=torch.bool)
    segment_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    _, mu, sigma = torch_util.get_running_stats(values, masks, segment_ids=segment_ids)

    expected_mu = torch.tensor([[[2.0, 6.0, 6.0, 5.5]]])
    expected_sigma = torch.tensor([[[1.0, 4.062019, 1.0, 1.118034]]])
    np.testing.assert_allclose(mu.numpy(), expected_mu.numpy(), atol=1e-6)
    np.testing.assert_allclose(sigma.numpy(), expected_sigma.numpy(), atol=1e-6)

  def test_get_running_stats_with_masks(self):
    values = torch.tensor([[[[1, 10, 3, 0], [100, 5, 200, 7]]]], dtype=torch.float32)
    masks = torch.tensor(
      [[[[False, True, False, True], [True, False, True, False]]]],
      dtype=torch.bool,
    )
    segment_ids = torch.tensor([[0, 0]], dtype=torch.int32)
    _, mu, sigma = torch_util.get_running_stats(values, masks, segment_ids=segment_ids)

    expected_mu = torch.tensor([[[2.0, 4.0]]])
    expected_sigma = torch.tensor([[[1.0, 2.236068]]])
    np.testing.assert_allclose(mu.numpy(), expected_mu.numpy(), atol=1e-6)
    np.testing.assert_allclose(sigma.numpy(), expected_sigma.numpy(), atol=1e-6)


if __name__ == "__main__":
  unittest.main()
