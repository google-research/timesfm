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

"""Tests for NaN-handling and interpolation utilities in the base module.

``strip_leading_nans`` and ``linear_interpolation`` sit on the critical
inference path: every user input passes through them before being
patched and fed to the transformer.  Incorrect behavior here — silently
keeping NaN values or interpolating the wrong indices — causes NaN
propagation through the entire model and produces garbage forecasts.
"""

import numpy as np

from timesfm.timesfm_2p5.timesfm_2p5_base import (
  linear_interpolation,
  moving_average,
  strip_leading_nans,
)


# ---------------------------------------------------------------------------
# strip_leading_nans
# ---------------------------------------------------------------------------


class TestStripLeadingNans:
  """Tests for strip_leading_nans — removes leading NaN prefix."""

  def test_no_nans_returns_unchanged(self):
    """An array without NaN values must pass through unmodified."""
    arr = np.array([1.0, 2.0, 3.0])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, arr)

  def test_strips_leading_nans_only(self):
    """Leading NaNs are removed; NaNs embedded in the middle are kept."""
    arr = np.array([np.nan, np.nan, 1.0, np.nan, 3.0])
    result = strip_leading_nans(arr)
    expected = np.array([1.0, np.nan, 3.0])
    np.testing.assert_array_equal(result, expected)

  def test_single_leading_nan(self):
    """Edge case: exactly one leading NaN."""
    arr = np.array([np.nan, 5.0, 6.0])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, np.array([5.0, 6.0]))

  def test_no_leading_nan_with_internal_nans(self):
    """If the first element is valid, nothing is stripped regardless of
    internal NaNs."""
    arr = np.array([1.0, np.nan, np.nan, 4.0])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, arr)

  def test_single_valid_element(self):
    """A single non-NaN element must be returned as-is."""
    arr = np.array([42.0])
    result = strip_leading_nans(arr)
    np.testing.assert_array_equal(result, np.array([42.0]))

  def test_all_nans_returns_full_array(self):
    """When every element is NaN, ``np.argmax`` on an all-False mask
    returns 0 — so the implementation returns the original array, not an
    empty one.

    This documents the *actual* behavior (which differs from the
    docstring claim of returning an empty array). Downstream code
    (``linear_interpolation``) is designed to handle this case.
    """
    arr = np.array([np.nan, np.nan, np.nan])
    result = strip_leading_nans(arr)
    # Actual behavior: argmax(~isnan) = 0 when all NaN → returns full array.
    assert len(result) == 3
    assert np.all(np.isnan(result))

  def test_preserves_dtype(self):
    """Output dtype must match input dtype (float32 stays float32)."""
    arr = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
    result = strip_leading_nans(arr)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# linear_interpolation
# ---------------------------------------------------------------------------


class TestLinearInterpolation:
  """Tests for linear_interpolation — fills NaN gaps via ``np.interp``."""

  def test_no_nans_returns_identical(self):
    """Without NaN values the array is returned as-is (fast path)."""
    arr = np.array([1.0, 2.0, 3.0])
    result = linear_interpolation(arr.copy())
    np.testing.assert_array_equal(result, arr)

  def test_interpolates_single_interior_nan(self):
    """A single interior NaN is linearly interpolated from neighbors."""
    arr = np.array([0.0, np.nan, 2.0])
    result = linear_interpolation(arr)
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0])

  def test_interpolates_multiple_interior_nans(self):
    """Multiple consecutive interior NaN values are interpolated."""
    arr = np.array([0.0, np.nan, np.nan, 3.0])
    result = linear_interpolation(arr)
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0])

  def test_extrapolates_trailing_nans(self):
    """Trailing NaN values are filled via ``np.interp`` which holds the
    last known value (nearest-neighbor extrapolation)."""
    arr = np.array([1.0, 2.0, np.nan, np.nan])
    result = linear_interpolation(arr)
    # np.interp extrapolates by clamping to boundary values.
    np.testing.assert_allclose(result, [1.0, 2.0, 2.0, 2.0])

  def test_extrapolates_leading_nans(self):
    """Leading NaN values are filled with the first valid value.

    In practice ``strip_leading_nans`` runs first, but the function must
    still be robust on its own.
    """
    arr = np.array([np.nan, np.nan, 3.0, 4.0])
    result = linear_interpolation(arr)
    np.testing.assert_allclose(result, [3.0, 3.0, 3.0, 4.0])

  def test_output_has_no_nans(self):
    """After interpolation, no NaN values should remain."""
    arr = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    result = linear_interpolation(arr)
    assert not np.any(np.isnan(result))

  def test_preserves_non_nan_values(self):
    """Non-NaN values in the original array must never be modified."""
    arr = np.array([10.0, np.nan, 30.0, np.nan, 50.0])
    original_valid = arr[~np.isnan(arr)].copy()
    result = linear_interpolation(arr)
    np.testing.assert_array_equal(
      result[~np.isnan(np.array([10.0, np.nan, 30.0, np.nan, 50.0]))],
      original_valid,
    )

  def test_interpolation_is_monotone_for_monotone_input(self):
    """If the known values are strictly increasing, the interpolated
    result must also be non-decreasing — a basic sanity check on the
    interpolation direction."""
    arr = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
    result = linear_interpolation(arr)
    diffs = np.diff(result)
    assert np.all(diffs >= 0)

  def test_single_non_nan_fills_all_gaps(self):
    """With only one valid value, every NaN is replaced by that value
    (np.interp clamps to the single known point)."""
    arr = np.array([np.nan, 5.0, np.nan])
    result = linear_interpolation(arr)
    np.testing.assert_allclose(result, [5.0, 5.0, 5.0])


# ---------------------------------------------------------------------------
# moving_average — decomposed forecasting helper
# ---------------------------------------------------------------------------


class TestMovingAverage:
  """Tests for moving_average — decomposes a series into trend and residual."""

  def test_constant_series(self):
    """For a constant series: after the first window_size elements,
    trend = constant, residual = 0."""
    arr = np.array([3.0] * 10)
    trend, residual = moving_average(arr, 3)
    # First (window_size-1) MA elements are distorted by zero-padding
    np.testing.assert_allclose(trend[2:], 3.0)
    np.testing.assert_allclose(residual[2:], 0.0, atol=1e-15)

  def test_sum_equals_original(self):
    """Trend + residual = original series (invariant)."""
    arr = np.array([1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 10.0])
    trend, residual = moving_average(arr, 3)
    np.testing.assert_allclose(trend + residual, arr)

  def test_trend_is_smoother(self):
    """Trend is smoother: max diff is less than original series."""
    arr = np.array([1.0, 10.0, 1.0, 10.0, 1.0, 10.0] * 10)
    trend, _ = moving_average(arr, 5)
    assert np.max(np.abs(np.diff(trend))) < np.max(np.abs(np.diff(arr)))

  def test_window_size_one(self):
    """When window_size=1, trend equals original series, residual = 0."""
    arr = np.array([1.0, 3.0, 2.0, 5.0])
    trend, residual = moving_average(arr, 1)
    np.testing.assert_allclose(trend, arr)
    np.testing.assert_allclose(residual, 0.0, atol=1e-15)

  def test_window_size_larger_than_array(self):
    """When window_size >= len(arr), the actual array length is used."""
    arr = np.array([1.0, 2.0, 3.0])
    trend, residual = moving_average(arr, 10)
    np.testing.assert_allclose(trend + residual, arr)
    assert len(trend) == len(arr)

  def test_empty_array(self):
    """Empty array returns two empty arrays."""
    arr = np.array([], dtype=np.float64)
    trend, residual = moving_average(arr, 3)
    assert len(trend) == 0
    assert len(residual) == 0

  def test_output_dtype(self):
    """Output is float64 (internal MA representation)."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    trend, residual = moving_average(arr, 2)
    assert trend.dtype == np.float64
    assert residual.dtype == np.float64

  def test_trend_residual_sum_white_noise(self):
    """For white noise: MA smoothing weakens trend component."""
    rng = np.random.RandomState(42)
    arr = rng.randn(100).astype(np.float64) * 2 + 5
    trend, residual = moving_average(arr, 10)
    np.testing.assert_allclose(trend + residual, arr)

  def test_actual_moving_average_values(self):
    """Check specific trend values for a given known series."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    trend, _ = moving_average(arr, 2)
    # MA(2) with one zero left-padding:
    # [0, 1, 2, 3, 4, 5] convolve [0.5, 0.5] → [0.5, 1.5, 2.5, 3.5, 4.5]
    np.testing.assert_allclose(trend, [0.5, 1.5, 2.5, 3.5, 4.5])
