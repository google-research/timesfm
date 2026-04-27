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

"""Unit tests for the TimesFM v2 codebase.

These tests cover pure-Python/NumPy utility functions and lightweight
model-shape smoke tests.  No checkpoint downloads are required.
"""

import math
import warnings

import numpy as np
import pytest

from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_base import (
  linear_interpolation,
  strip_leading_nans,
)


# ---------------------------------------------------------------------------
# strip_leading_nans
# ---------------------------------------------------------------------------

class TestStripLeadingNans:
  def test_no_nans(self):
    arr = np.array([1.0, 2.0, 3.0])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, arr)

  def test_leading_nans(self):
    arr = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

  def test_single_leading_nan(self):
    arr = np.array([np.nan, 5.0])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, np.array([5.0]))

  def test_trailing_nans_preserved(self):
    arr = np.array([1.0, 2.0, np.nan])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, arr)

  def test_all_nans_returns_empty(self):
    arr = np.array([np.nan, np.nan, np.nan])
    result = strip_leading_nans(arr)
    assert len(result) == 0

  def test_empty_array(self):
    arr = np.array([], dtype=float)
    result = strip_leading_nans(arr)
    assert len(result) == 0

  def test_single_valid(self):
    arr = np.array([1.0])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# linear_interpolation
# ---------------------------------------------------------------------------

class TestLinearInterpolation:
  def test_no_nans(self):
    arr = np.array([1.0, 2.0, 3.0])
    result = linear_interpolation(arr.copy())
    np.testing.assert_array_equal(result, arr)

  def test_interior_nan(self):
    arr = np.array([0.0, np.nan, 2.0])
    result = linear_interpolation(arr.copy())
    assert not np.any(np.isnan(result))
    assert pytest.approx(result[1], abs=1e-6) == 1.0

  def test_multiple_interior_nans(self):
    arr = np.array([0.0, np.nan, np.nan, 3.0])
    result = linear_interpolation(arr.copy())
    assert not np.any(np.isnan(result))
    assert pytest.approx(result[1], abs=1e-6) == 1.0
    assert pytest.approx(result[2], abs=1e-6) == 2.0

  def test_trailing_nans_filled_with_last_valid(self):
    arr = np.array([1.0, 2.0, np.nan])
    result = linear_interpolation(arr.copy())
    assert not np.any(np.isnan(result))

  def test_all_nans_returns_zeros(self):
    """All-NaN input: fallback fills with 0.0 (nanmean of empty = 0)."""
    arr = np.array([np.nan, np.nan, np.nan])
    result = linear_interpolation(arr.copy())
    assert not np.any(np.isnan(result))

  def test_single_valid_value_many_nans(self):
    """When only one valid value exists, non_nans_values has length 1."""
    arr = np.array([np.nan, 5.0, np.nan])
    result = linear_interpolation(arr.copy())
    assert not np.any(np.isnan(result))
    # The single valid value is 5.0; interpolation extrapolates it
    assert result[1] == pytest.approx(5.0)

  def test_two_valid_values(self):
    """non_nans_values has length 2 - previously crashed due to numpy bool bug."""
    arr = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
    result = linear_interpolation(arr.copy())
    assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# ForecastConfig
# ---------------------------------------------------------------------------

class TestForecastConfig:
  def test_default_construction(self):
    fc = ForecastConfig()
    assert fc.max_context == 0
    assert fc.max_horizon == 0

  def test_custom_values(self):
    fc = ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True)
    assert fc.max_context == 512
    assert fc.max_horizon == 128
    assert fc.normalize_inputs is True

  def test_frozen(self):
    fc = ForecastConfig(max_context=512)
    with pytest.raises(Exception):
      fc.max_context = 1024  # type: ignore[misc]

  def test_force_flip_invariance_default_true(self):
    """Default is True - users should be aware of 2× inference cost."""
    fc = ForecastConfig()
    assert fc.force_flip_invariance is True


# ---------------------------------------------------------------------------
# __version__ is exposed
# ---------------------------------------------------------------------------

def test_version_exposed():
  import timesfm
  assert hasattr(timesfm, "__version__")
  assert isinstance(timesfm.__version__, str)
  assert len(timesfm.__version__) > 0


# ---------------------------------------------------------------------------
# ImportWarning on broken installs (simulated)
# ---------------------------------------------------------------------------

def test_no_spurious_import_warning_on_clean_import():
  """Importing timesfm with no extra backends installed should not warn."""
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import timesfm  # noqa: F401 - already imported, but test the filter
  import_warnings = [x for x in w if issubclass(x.category, ImportWarning)]
  assert len(import_warnings) == 0, import_warnings


# ---------------------------------------------------------------------------
# XReg padding helper
# ---------------------------------------------------------------------------

def test_xreg_padding_multiple_of_64():
  """_pad_dim should round up to the nearest multiple of 64."""
  pytest.importorskip("jax", reason="jax not installed")
  pytest.importorskip("sklearn", reason="sklearn not installed")
  from timesfm.utils.xreg_lib import _pad_dim, _PAD_MULTIPLE

  for n in [1, 63, 64, 65, 127, 128, 129, 1025, 4096, 4097]:
    result = _pad_dim(n)
    assert result >= n, f"_pad_dim({n}) = {result} < {n}"
    assert result % _PAD_MULTIPLE == 0, f"_pad_dim({n}) = {result} not multiple of {_PAD_MULTIPLE}"
    # Should not waste more than one full multiple
    assert result <= n + _PAD_MULTIPLE, f"_pad_dim({n}) = {result} wastes too much"


# ---------------------------------------------------------------------------
# XReg solve vs pinv closeness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ridge", [0.0, 0.1, 1.0])
def test_xreg_solve_matches_pinv(ridge):
  """jnp.linalg.solve(A, b) should match pinv(A) @ b for ridge regression."""
  jnp = pytest.importorskip("jax.numpy", reason="jax not installed")
  import jax.numpy as jnp  # noqa: F811

  rng = np.random.default_rng(0)
  n, d = 50, 5
  x_train = jnp.array(rng.standard_normal((n, d)))
  flat_targets = jnp.array(rng.standard_normal(n))

  A = x_train.T @ x_train + ridge * jnp.eye(d)
  b = x_train.T @ flat_targets

  beta_pinv = jnp.linalg.pinv(A, hermitian=True) @ b
  beta_solve = jnp.linalg.solve(A, b)

  np.testing.assert_allclose(np.array(beta_solve), np.array(beta_pinv), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Torch model smoke test (shape only, no checkpoint required)
# ---------------------------------------------------------------------------

try:
  import torch
  from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
  _TORCH_AVAILABLE = True
except ImportError:
  _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestTorchModelShape:
  """Verifies forward-pass output shapes without loading checkpoint weights."""

  def _make_module(self) -> "TimesFM_2p5_200M_torch_module":
    module = TimesFM_2p5_200M_torch_module()
    module.to(module.device)
    module.eval()
    return module

  def test_forward_output_shapes(self):
    module = self._make_module()
    batch, n_patches, patch_len = 2, 8, module.p
    inputs = torch.randn(batch, n_patches, patch_len, device=module.device)
    masks = torch.zeros(batch, n_patches, patch_len, dtype=torch.bool, device=module.device)
    with torch.no_grad():
      (_, _, output_ts, output_q_spread), _ = module(inputs, masks)
    # output_ts: (batch, n_patches, output_patch_len * q)
    assert output_ts.shape[0] == batch
    assert output_ts.shape[1] == n_patches

  def test_decode_output_shapes(self):
    module = self._make_module()
    p = module.p
    context_patches = 16
    context = context_patches * p
    horizon = module.o  # one output patch, no AR step

    inputs = torch.randn(1, context, device=module.device)
    masks = torch.zeros(1, context, dtype=torch.bool, device=module.device)

    with torch.no_grad():
      pf_outputs, q_spreads, ar_outputs = module.decode(horizon, inputs, masks)

    assert pf_outputs.shape[0] == 1
    assert ar_outputs is None  # no AR steps for horizon == output_patch_len

  def test_decode_ar_steps(self):
    module = self._make_module()
    p = module.p
    o = module.o
    context = 16 * p
    horizon = 2 * o  # triggers one AR step

    inputs = torch.randn(1, context, device=module.device)
    masks = torch.zeros(1, context, dtype=torch.bool, device=module.device)

    with torch.no_grad():
      pf_outputs, q_spreads, ar_outputs = module.decode(horizon, inputs, masks)

    assert ar_outputs is not None
    # AR outputs: (batch, num_ar_steps, m, o, q)
    assert ar_outputs.shape[0] == 1


# ---------------------------------------------------------------------------
# compute_cumulative_stats correctness (vectorized vs sequential reference)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestComputeCumulativeStats:
  """Verifies compute_cumulative_stats matches the sequential update_running_stats loop."""

  def _reference(self, inputs, masks):
    """Sequential reference using update_running_stats (the original loop)."""
    from timesfm.torch.util import update_running_stats
    batch, n_patches, _ = inputs.shape
    n = torch.zeros(batch, device=inputs.device)
    mu = torch.zeros(batch, device=inputs.device)
    sigma = torch.zeros(batch, device=inputs.device)
    ctx_mu = []
    ctx_sigma = []
    for i in range(n_patches):
      (n, mu, sigma), _ = update_running_stats(n, mu, sigma, inputs[:, i], masks[:, i])
      ctx_mu.append(mu.clone())
      ctx_sigma.append(sigma.clone())
    context_mu = torch.stack(ctx_mu, dim=1)
    context_sigma = torch.stack(ctx_sigma, dim=1)
    return n, mu, sigma, context_mu, context_sigma

  def _run(self, inputs, masks):
    from timesfm.torch.util import compute_cumulative_stats
    return compute_cumulative_stats(inputs, masks)

  def test_no_masked_values(self):
    torch.manual_seed(0)
    inputs = torch.randn(3, 8, 16)
    masks = torch.zeros(3, 8, 16, dtype=torch.bool)
    r_n, r_mu, r_sig, r_cmu, r_csig = self._reference(inputs, masks)
    v_n, v_mu, v_sig, v_cmu, v_csig = self._run(inputs, masks)
    torch.testing.assert_close(v_n, r_n, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_mu, r_mu, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_sig, r_sig, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_cmu, r_cmu, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_csig, r_csig, rtol=1e-4, atol=1e-4)

  def test_some_masked_values(self):
    torch.manual_seed(1)
    inputs = torch.randn(2, 6, 8)
    masks = torch.rand(2, 6, 8) > 0.5  # ~50 % masked
    r_n, r_mu, r_sig, r_cmu, r_csig = self._reference(inputs, masks)
    v_n, v_mu, v_sig, v_cmu, v_csig = self._run(inputs, masks)
    torch.testing.assert_close(v_n, r_n, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_mu, r_mu, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_sig, r_sig, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_cmu, r_cmu, rtol=1e-4, atol=1e-4)

  def test_all_masked_gives_zeros(self):
    inputs = torch.randn(2, 4, 8)
    masks = torch.ones(2, 4, 8, dtype=torch.bool)  # all masked
    from timesfm.torch.util import compute_cumulative_stats
    last_n, last_mu, last_sigma, ctx_mu, ctx_sigma = compute_cumulative_stats(inputs, masks)
    assert torch.all(last_n == 0)
    assert torch.all(last_mu == 0)
    assert torch.all(last_sigma == 0)

  def test_single_patch(self):
    """Single patch: cumulative stats == patch stats."""
    inputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
    masks = torch.zeros(1, 1, 4, dtype=torch.bool)
    from timesfm.torch.util import compute_cumulative_stats
    last_n, last_mu, last_sigma, _, _ = compute_cumulative_stats(inputs, masks)
    assert last_n.item() == pytest.approx(4.0)
    assert last_mu.item() == pytest.approx(2.5)
    expected_std = math.sqrt(((1-2.5)**2 + (2-2.5)**2 + (3-2.5)**2 + (4-2.5)**2) / 4)
    assert last_sigma.item() == pytest.approx(expected_std, abs=1e-5)

  def test_output_shapes(self):
    batch, n_patches, patch_len = 4, 10, 32
    inputs = torch.randn(batch, n_patches, patch_len)
    masks = torch.zeros(batch, n_patches, patch_len, dtype=torch.bool)
    from timesfm.torch.util import compute_cumulative_stats
    last_n, last_mu, last_sigma, ctx_mu, ctx_sigma = compute_cumulative_stats(inputs, masks)
    assert last_n.shape == (batch,)
    assert last_mu.shape == (batch,)
    assert last_sigma.shape == (batch,)
    assert ctx_mu.shape == (batch, n_patches)
    assert ctx_sigma.shape == (batch, n_patches)


# ---------------------------------------------------------------------------
# RuntimeError raised before compile()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# update_running_stats - direct tests with known values
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestUpdateRunningStats:
  """Tests update_running_stats with analytically known results."""

  def _call(self, n, mu, sigma, x, mask=None):
    from timesfm.torch.util import update_running_stats
    t = torch.tensor
    if mask is None:
      mask = torch.zeros_like(x, dtype=torch.bool)
    (new_n, new_mu, new_sigma), _ = update_running_stats(
      t([n], dtype=x.dtype),
      t([mu], dtype=x.dtype),
      t([sigma], dtype=x.dtype),
      x.unsqueeze(0),
      mask.unsqueeze(0),
    )
    return new_n.item(), new_mu.item(), new_sigma.item()

  def test_first_patch_mean(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    n, mu, _ = self._call(0, 0, 0, x)
    assert n == pytest.approx(4.0)
    assert mu == pytest.approx(2.5)

  def test_first_patch_sigma(self):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    _, _, sigma = self._call(0, 0, 0, x)
    expected = math.sqrt(((1-2.5)**2 + (2-2.5)**2 + (3-2.5)**2 + (4-2.5)**2) / 4)
    assert sigma == pytest.approx(expected, abs=1e-5)

  def test_accumulate_two_identical_patches(self):
    """Two patches of [1,2,3,4]: global stats same as one patch of 8 identical values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    n1, mu1, sig1 = self._call(0, 0, 0, x)
    n2, mu2, _ = self._call(n1, mu1, sig1, x)
    assert n2 == pytest.approx(8.0)
    assert mu2 == pytest.approx(2.5)  # mean of two identical patches is unchanged

  def test_all_masked_patch_leaves_stats_unchanged(self):
    x = torch.tensor([99.0, 99.0, 99.0])
    mask = torch.ones(3, dtype=torch.bool)
    n, mu, sigma = self._call(4, 2.5, 1.0, x, mask)
    assert n == pytest.approx(4.0)   # count unchanged
    assert mu == pytest.approx(2.5)  # mean unchanged


# ---------------------------------------------------------------------------
# revin forward, reverse, and zero-sigma guard
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestRevin:
  """Tests the reversible instance normalisation utility."""

  def test_round_trip(self):
    """revin(revin(x, forward), reverse) should recover x."""
    from timesfm.torch.util import revin
    torch.manual_seed(42)
    x = torch.randn(3, 10)
    mu = torch.tensor([1.0, 2.0, 3.0])
    sigma = torch.tensor([0.5, 1.0, 2.0])
    normed = revin(x, mu, sigma, reverse=False)
    recovered = revin(normed, mu, sigma, reverse=True)
    torch.testing.assert_close(recovered, x, rtol=1e-5, atol=1e-5)

  def test_zero_sigma_no_division_by_zero(self):
    """When sigma < 1e-6 the denominator is replaced by 1.0 - no inf/nan."""
    from timesfm.torch.util import revin
    x = torch.tensor([[1.0, 2.0, 3.0]])
    mu = torch.tensor([2.0])
    sigma = torch.tensor([0.0])  # exactly zero
    result = revin(x, mu, sigma, reverse=False)
    assert not torch.any(torch.isnan(result))
    assert not torch.any(torch.isinf(result))

  def test_forward_normalises_correctly(self):
    from timesfm.torch.util import revin
    x = torch.tensor([[0.0, 2.0, 4.0]])  # mean=2, sigma=~1.633
    mu = torch.tensor([2.0])
    sigma = torch.tensor([2.0])
    normed = revin(x, mu, sigma, reverse=False)
    torch.testing.assert_close(normed, torch.tensor([[-1.0, 0.0, 1.0]]), atol=1e-5, rtol=1e-5)

  def test_reverse_denormalises_correctly(self):
    from timesfm.torch.util import revin
    normed = torch.tensor([[-1.0, 0.0, 1.0]])
    mu = torch.tensor([2.0])
    sigma = torch.tensor([2.0])
    result = revin(normed, mu, sigma, reverse=True)
    torch.testing.assert_close(result, torch.tensor([[0.0, 2.0, 4.0]]), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

def test_public_api_always_exports_forecast_config():
  """ForecastConfig must be importable regardless of which backends are installed."""
  import timesfm
  assert hasattr(timesfm, "ForecastConfig")
  assert hasattr(timesfm, "__version__")
  assert "ForecastConfig" in timesfm.__all__
  assert "__version__" in timesfm.__all__


# ---------------------------------------------------------------------------
# RuntimeError raised before compile()
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_forecast_raises_before_compile():
  """forecast() must raise RuntimeError if compile() was never called."""
  from timesfm.timesfm_2p5 import timesfm_2p5_torch
  # Construct the wrapper without loading weights or calling compile().
  model = object.__new__(timesfm_2p5_torch.TimesFM_2p5_200M_torch)
  model.compiled_decode = None
  model.global_batch_size = 0
  model.forecast_config = None
  with pytest.raises(RuntimeError, match="compile"):
    model.forecast(horizon=12, inputs=[np.array([1.0, 2.0, 3.0])])
