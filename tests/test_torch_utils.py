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

"""Tests for PyTorch utility functions: running statistics and RevIN.

These utilities are invoked at every patch boundary during autoregressive
decoding. Bugs here cause silent numerical drift that compounds over
long horizons, making them especially hard to diagnose from forecast
output alone.
"""

import torch
import numpy as np
import pytest

from timesfm.torch.util import (
  DecodeCache,
  _TOLERANCE,
  revin,
  update_running_stats,
)


# ---------------------------------------------------------------------------
# update_running_stats
# ---------------------------------------------------------------------------


class TestUpdateRunningStats:
  """Tests for Welford-style online mean / variance accumulation."""

  def test_single_batch_matches_numpy(self):
    """A single update with no mask must match numpy's mean and std.

    This is the most basic correctness check: feed all values at once
    and compare against the ground-truth statistics.
    """
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    mask = torch.zeros_like(x, dtype=torch.bool)
    n0 = torch.zeros(1)
    mu0 = torch.zeros(1)
    sigma0 = torch.zeros(1)

    (new_n, new_mu, new_sigma), _ = update_running_stats(n0, mu0, sigma0, x, mask)

    np_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_mu = np.mean(np_values)
    # Population std (ddof=0), same as PyTorch default.
    expected_sigma = np.std(np_values, ddof=0)

    assert new_n.item() == pytest.approx(5.0)
    assert new_mu.item() == pytest.approx(expected_mu, abs=1e-5)
    assert new_sigma.item() == pytest.approx(expected_sigma, abs=1e-5)

  def test_incremental_accumulation_matches_full_computation(self):
    """Accumulating two batches incrementally must yield the same result
    as computing statistics over all values at once.

    This is the defining property of online/streaming statistics.
    """
    all_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    batch1 = torch.tensor([[1.0, 2.0, 3.0]])
    batch2 = torch.tensor([[4.0, 5.0, 6.0]])

    def no_mask(t):
      return torch.zeros_like(t, dtype=torch.bool)

    # Full computation.
    n0 = torch.zeros(1)
    mu0 = torch.zeros(1)
    sigma0 = torch.zeros(1)
    (full_n, full_mu, full_sigma), _ = update_running_stats(
      n0, mu0, sigma0, all_values, no_mask(all_values)
    )

    # Incremental computation.
    (n1, mu1, sigma1), _ = update_running_stats(
      n0, mu0, sigma0, batch1, no_mask(batch1)
    )
    (inc_n, inc_mu, inc_sigma), _ = update_running_stats(
      n1, mu1, sigma1, batch2, no_mask(batch2)
    )

    assert inc_n.item() == pytest.approx(full_n.item())
    assert inc_mu.item() == pytest.approx(full_mu.item(), abs=1e-5)
    assert inc_sigma.item() == pytest.approx(full_sigma.item(), abs=1e-5)

  def test_masked_elements_excluded_from_statistics(self):
    """Masked positions must be completely ignored — as if they don't exist.

    In TimesFM, leading padding is masked. If mask handling is broken,
    the zero-padding values pollute the running mean and variance.
    """
    # Two values: 10 and 20 are valid; 0 is masked.
    x = torch.tensor([[0.0, 10.0, 20.0]])
    mask = torch.tensor([[True, False, False]])
    n0 = torch.zeros(1)
    mu0 = torch.zeros(1)
    sigma0 = torch.zeros(1)

    (new_n, new_mu, new_sigma), _ = update_running_stats(n0, mu0, sigma0, x, mask)

    assert new_n.item() == pytest.approx(2.0)
    assert new_mu.item() == pytest.approx(15.0, abs=1e-5)
    expected_sigma = np.std([10.0, 20.0], ddof=0)
    assert new_sigma.item() == pytest.approx(expected_sigma, abs=1e-5)

  def test_all_masked_yields_zero_stats(self):
    """When every element is masked, the function must return zeros
    rather than NaN or raise an error.

    This happens when an input series is entirely padding.
    """
    x = torch.tensor([[99.0, 99.0, 99.0]])
    mask = torch.ones_like(x, dtype=torch.bool)
    n0 = torch.zeros(1)
    mu0 = torch.zeros(1)
    sigma0 = torch.zeros(1)

    (new_n, new_mu, new_sigma), _ = update_running_stats(n0, mu0, sigma0, x, mask)

    assert new_n.item() == 0.0
    assert new_mu.item() == 0.0
    assert new_sigma.item() == 0.0

  def test_batched_computation_independent(self):
    """Each sample in the batch must be computed independently.

    Cross-sample leakage would corrupt multi-series forecasting.
    """
    x = torch.tensor(
      [
        [1.0, 2.0, 3.0],
        [100.0, 200.0, 300.0],
      ]
    )
    mask = torch.zeros_like(x, dtype=torch.bool)
    n0 = torch.zeros(2)
    mu0 = torch.zeros(2)
    sigma0 = torch.zeros(2)

    (new_n, new_mu, new_sigma), _ = update_running_stats(n0, mu0, sigma0, x, mask)

    assert new_mu[0].item() == pytest.approx(2.0, abs=1e-5)
    assert new_mu[1].item() == pytest.approx(200.0, abs=1e-5)

    expected_sigma_0 = np.std([1.0, 2.0, 3.0], ddof=0)
    expected_sigma_1 = np.std([100.0, 200.0, 300.0], ddof=0)
    assert new_sigma[0].item() == pytest.approx(expected_sigma_0, abs=1e-5)
    assert new_sigma[1].item() == pytest.approx(expected_sigma_1, abs=1e-5)

  def test_constant_input_yields_zero_sigma(self):
    """A constant series has zero variance — sigma must be exactly 0.

    This is important because ``revin`` guards against division-by-zero
    using ``_TOLERANCE`` when sigma is near zero.
    """
    x = torch.tensor([[7.0, 7.0, 7.0, 7.0]])
    mask = torch.zeros_like(x, dtype=torch.bool)
    n0 = torch.zeros(1)
    mu0 = torch.zeros(1)
    sigma0 = torch.zeros(1)

    (_, new_mu, new_sigma), _ = update_running_stats(n0, mu0, sigma0, x, mask)

    assert new_mu.item() == pytest.approx(7.0)
    assert new_sigma.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# revin (Reversible Instance Normalization)
# ---------------------------------------------------------------------------


class TestRevIN:
  """Tests for the RevIN normalization used in patched decoding."""

  def test_forward_then_reverse_is_identity(self):
    """normalize → denormalize must reconstruct the original tensor.

    This is the fundamental invariant of reversible normalization: the
    model operates in normalized space, and the output is mapped back
    to the original scale. Any deviation here directly corrupts the
    final forecast values.
    """
    x = torch.tensor([[10.0, 20.0, 30.0]])
    mu = torch.tensor([20.0])
    sigma = torch.tensor([10.0])

    normed = revin(x, mu, sigma, reverse=False)
    recovered = revin(normed, mu, sigma, reverse=True)

    torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)

  def test_forward_produces_correct_normalization(self):
    """After forward normalization: (x - mu) / sigma."""
    x = torch.tensor([[10.0, 20.0, 30.0]])
    mu = torch.tensor([20.0])
    sigma = torch.tensor([10.0])

    normed = revin(x, mu, sigma, reverse=False)

    expected = torch.tensor([[-1.0, 0.0, 1.0]])
    torch.testing.assert_close(normed, expected, atol=1e-5, rtol=1e-5)

  def test_reverse_produces_correct_denormalization(self):
    """After reverse: x * sigma + mu."""
    normed = torch.tensor([[-1.0, 0.0, 1.0]])
    mu = torch.tensor([20.0])
    sigma = torch.tensor([10.0])

    recovered = revin(normed, mu, sigma, reverse=True)

    expected = torch.tensor([[10.0, 20.0, 30.0]])
    torch.testing.assert_close(recovered, expected, atol=1e-5, rtol=1e-5)

  def test_zero_sigma_does_not_produce_nan(self):
    """When sigma < tolerance, the function substitutes 1.0 to avoid
    division by zero. This occurs for constant-valued input series.

    NaN propagation from here would poison the entire transformer
    forward pass.
    """
    x = torch.tensor([[5.0, 5.0, 5.0]])
    mu = torch.tensor([5.0])
    sigma = torch.tensor([0.0])  # zero variance

    normed = revin(x, mu, sigma, reverse=False)

    assert not torch.any(torch.isnan(normed))
    assert not torch.any(torch.isinf(normed))

  def test_near_zero_sigma_guarded_by_tolerance(self):
    """Sigma values just below ``_TOLERANCE`` must trigger the guard."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    mu = torch.tensor([2.0])
    sigma = torch.tensor([_TOLERANCE / 2])  # below threshold

    normed = revin(x, mu, sigma, reverse=False)

    assert not torch.any(torch.isnan(normed))
    # With sigma replaced by 1.0: result = x - mu
    expected = torch.tensor([[-1.0, 0.0, 1.0]])
    torch.testing.assert_close(normed, expected, atol=1e-5, rtol=1e-5)

  def test_roundtrip_with_batched_3d_input(self):
    """RevIN must broadcast correctly for (batch, patches, patch_len)
    tensors — the actual shape used during patched decoding."""
    batch, patches, patch_len = 2, 4, 32
    x = torch.randn(batch, patches, patch_len)
    mu = torch.tensor([1.0, 2.0])  # (batch,)
    sigma = torch.tensor([3.0, 4.0])  # (batch,)

    normed = revin(x, mu, sigma, reverse=False)
    recovered = revin(normed, mu, sigma, reverse=True)

    torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)

  def test_roundtrip_with_batched_4d_input(self):
    """RevIN must broadcast correctly for (batch, patches, patch_len, q)
    tensors — the shape used for quantile outputs.

    In the actual decode path, mu/sigma have shape (batch, patches) for
    4D tensors, so the ``len(mu.shape) == len(x.shape) - 2`` branch
    fires and adds two trailing singleton dimensions.
    """
    batch, patches, patch_len, q = 2, 4, 32, 10
    x = torch.randn(batch, patches, patch_len, q)
    # Match the actual call-site shape: (batch, patches)
    mu = torch.randn(batch, patches)
    sigma = torch.abs(torch.randn(batch, patches)) + 1.0  # ensure positive

    normed = revin(x, mu, sigma, reverse=False)
    recovered = revin(normed, mu, sigma, reverse=True)

    torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)

  def test_negative_values_handled_correctly(self):
    """RevIN must work for series with negative values (e.g. temperature,
    financial returns). ``infer_is_positive`` is a separate downstream
    flag and does not affect RevIN itself.
    """
    x = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])
    mu = torch.tensor([0.0])
    sigma = torch.tensor([7.07])

    normed = revin(x, mu, sigma, reverse=False)
    recovered = revin(normed, mu, sigma, reverse=True)

    torch.testing.assert_close(recovered, x, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# DecodeCache
# ---------------------------------------------------------------------------


class TestDecodeCache:
  """Tests for the DecodeCache dataclass used in KV-cache decoding."""

  def test_is_mutable(self):
    """DecodeCache is *not* frozen — the attention loop mutates
    ``next_index`` and ``num_masked`` in-place during autoregressive
    decoding."""
    cache = DecodeCache(
      next_index=torch.tensor([0]),
      num_masked=torch.tensor([0]),
      key=torch.zeros(1, 10, 4, 8),
      value=torch.zeros(1, 10, 4, 8),
    )
    cache.next_index = torch.tensor([5])
    assert cache.next_index.item() == 5

  def test_key_value_shape_consistency(self):
    """Key and value tensors must have identical shapes — they are
    indexed in parallel during attention computation."""
    batch, seq, heads, head_dim = 2, 64, 16, 80
    cache = DecodeCache(
      next_index=torch.zeros(batch, dtype=torch.int32),
      num_masked=torch.zeros(batch, dtype=torch.int32),
      key=torch.zeros(batch, seq, heads, head_dim),
      value=torch.zeros(batch, seq, heads, head_dim),
    )
    assert cache.key.shape == cache.value.shape
    assert cache.key.shape == (batch, seq, heads, head_dim)
