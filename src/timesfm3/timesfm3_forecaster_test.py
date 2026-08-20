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

"""Tests for PyTorch TimesFM3Forecaster."""

import unittest
from unittest import mock
import numpy as np
import torch

from . import timesfm3_forecaster


class FakeModel(torch.nn.Module):
  """A tiny fake model for testing."""

  def __init__(self, input_patch_len=8, output_patch_len=8, quantiles=3):
    super().__init__()
    self.input_patch_len = input_patch_len
    self.output_patch_len = output_patch_len
    self.quantiles = quantiles

  def decode(
      self,
      target,
      autoregressive_index=0,
      horizon=8,
      past_only_covariates=None,
      past_future_covariates=None,
      mask=None,
  ):
    del autoregressive_index, past_only_covariates, past_future_covariates, mask
    b, v, _ = target.shape
    shape = (b, v, horizon, self.quantiles)
    return torch.ones(shape, dtype=torch.float32) * 2.0


class TimesFM3ForecasterTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = timesfm3_forecaster._ModelConfig(
        checkpoint_path="/fake/path",
        per_core_batch_size=2,
        input_patch_length=8,
        output_patch_length=8,
        median_quantile_index=1,
    )

  def test_strip_leading_nans_1d(self):
    arr = np.array([np.nan, np.nan, 1.0, 2.0, np.nan])
    res = timesfm3_forecaster.strip_leading_nans(arr)
    np.testing.assert_array_equal(res, np.array([1.0, 2.0, np.nan]))

  def test_strip_leading_nans_2d(self):
    arr = np.array([
        [np.nan, 1.0, 2.0],
        [np.nan, np.nan, 3.0],
    ])
    res = timesfm3_forecaster.strip_leading_nans(arr)
    expected = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
    ])
    np.testing.assert_array_equal(res, expected)

  def test_linear_interpolation(self):
    arr = np.array([1.0, np.nan, 3.0])
    res = timesfm3_forecaster.linear_interpolation(arr)
    np.testing.assert_array_equal(res, np.array([1.0, 2.0, 3.0]))

  def test_znorm_stats(self):
    arr = np.array([1.0, 2.0, 3.0])
    mu, sigma = timesfm3_forecaster._znorm_stats(arr)
    self.assertAlmostEqual(mu, 2.0)
    self.assertAlmostEqual(sigma, np.std([1.0, 2.0, 3.0]))

  def test_is_nonnegative(self):
    arr_pos = np.array([0.0, 1.0, 2.0])
    arr_neg = np.array([-1.0, 1.0, 2.0])
    self.assertTrue(timesfm3_forecaster._is_nonnegative(arr_pos))
    self.assertFalse(timesfm3_forecaster._is_nonnegative(arr_neg))

  @mock.patch.object(
      timesfm3_forecaster.TimesFM3Forecaster, "_init_model", autospec=True
  )
  def test_univariate_predict(self, _):
    forecaster = timesfm3_forecaster.TimesFM3Forecaster(self.config)
    forecaster.model = FakeModel()
    forecaster.device = torch.device("cpu")

    ctx = np.array([1.0, 2.0, 3.0, 4.0])
    out = forecaster.predict(ctx, horizon=4)
    self.assertEqual(out.forecast.shape, (4,))
    np.testing.assert_array_equal(out.forecast, np.full((4,), 2.0))

  @mock.patch.object(
      timesfm3_forecaster.TimesFM3Forecaster, "_init_model", autospec=True
  )
  def test_multivariate_predict_with_covariates(self, _):
    forecaster = timesfm3_forecaster.TimesFM3Forecaster(self.config)
    forecaster.model = FakeModel()
    forecaster.device = torch.device("cpu")

    ctx = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])
    po_cov = np.array([1.0, 1.0, 1.0, 1.0])
    pf_cov = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    out = forecaster.predict(
        ctx,
        horizon=4,
        past_only_covariates=po_cov,
        past_future_covariates=pf_cov,
        return_quantiles=True,
    )
    self.assertEqual(out.forecast.shape, (2, 4))
    self.assertEqual(out.quantiles.shape, (2, 4, 3))


if __name__ == "__main__":
  unittest.main()
