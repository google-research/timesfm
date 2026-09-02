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

"""Tests for PyTorch TimesFM3Forecaster."""

import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import configs, timesfm3_forecaster
from . import model as torch_model_lib


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

  def test_linear_interpolation(self):
    arr = np.array([1.0, np.nan, 3.0])
    res = timesfm3_forecaster.linear_interpolation(arr)
    np.testing.assert_array_equal(res, np.array([1.0, 2.0, 3.0]))

  def test_linear_interpolation_all_nan(self):
    arr = np.array([np.nan, np.nan, np.nan])
    res = timesfm3_forecaster.linear_interpolation(arr)
    np.testing.assert_array_equal(res, np.array([0.0, 0.0, 0.0]))

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

    ctx = np.array(
      [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
      ]
    )
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

  def test_query_format_padding_and_truncation(self):
    # Test padding when context_length < context_len
    q_short = timesfm3_forecaster._Query(
      horizon=4,
      targets=np.array([[1.0, 2.0]]),
      past_only_covariates=np.array([[5.0, 6.0]]),
      past_future_covariates=np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )
    hor, tgt, mask, po, pf = q_short.format(context_len=4)
    self.assertEqual(hor, 4)
    self.assertEqual(tgt.shape, (1, 4))
    np.testing.assert_array_equal(tgt, np.array([[0.0, 0.0, 1.0, 2.0]]))
    np.testing.assert_array_equal(mask, np.array([True, True, False, False]))
    np.testing.assert_array_equal(po, np.array([[0.0, 0.0, 5.0, 6.0]]))
    self.assertEqual(pf.shape, (1, 8))  # 4 ctx + 4 horizon

    # Test truncation when context_length > context_len
    q_long = timesfm3_forecaster._Query(
      horizon=2,
      targets=np.array([[1.0, 2.0, 3.0, 4.0]]),
      past_only_covariates=np.array([[10.0, 20.0, 30.0, 40.0]]),
      past_future_covariates=np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )
    hor, tgt, mask, po, pf = q_long.format(context_len=2)
    self.assertEqual(hor, 2)
    self.assertEqual(tgt.shape, (1, 2))
    np.testing.assert_array_equal(tgt, np.array([[3.0, 4.0]]))
    np.testing.assert_array_equal(mask, np.array([False, False]))
    np.testing.assert_array_equal(po, np.array([[30.0, 40.0]]))

  def test_from_pretrained_local_directory(self):
    resblock_config = configs.ResidualBlockConfig(
      hidden_dims=16,
      output_dims=16,
      use_bias=False,
      activation="relu",
    )
    transformer_config = configs.StackedTransformersConfig(
      num_layers=1,
      transformer=configs.TransformerConfig(
        model_dims=16,
        hidden_dims=16,
        num_heads=2,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_rope_seq=True,
        use_rope_var=False,
        use_bias=False,
        ff_activation="relu",
        deterministic=True,
      ),
    )
    model = torch_model_lib.TimesFM3Torch(
      input_patch_len=8,
      output_patch_len=16,
      quantiles=[0.1, 0.5, 0.9],
      residual_block_config=resblock_config,
      transformer_config=transformer_config,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      model.save_pretrained(tmpdir)
      forecaster = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(
        tmpdir, device="cpu"
      )
      self.assertEqual(forecaster.config.input_patch_length, 8)
      self.assertEqual(forecaster.config.output_patch_length, 16)
      ctx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      out = forecaster.predict(ctx, horizon=8)
      self.assertEqual(out.forecast.shape, (8,))


class _RecordingFakeModel(FakeModel):
  """FakeModel that records decode inputs and infers horizon like the real model."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.calls = []

  def decode(
    self,
    target,
    autoregressive_index=0,
    horizon=8,
    past_only_covariates=None,
    past_future_covariates=None,
    mask=None,
  ):
    if past_future_covariates is not None:
      horizon = past_future_covariates.shape[-1] - target.shape[-1]
    self.calls.append(
      dict(
        target=target.clone(),
        past_future_covariates=(
          None if past_future_covariates is None else past_future_covariates.clone()
        ),
        horizon=horizon,
      )
    )
    return super().decode(target, autoregressive_index, horizon, mask=mask)


class TimesFM3ForecasterCovariateWindowTest(unittest.TestCase):
  """Regression tests for past-future covariate windows under truncation."""

  def setUp(self):
    super().setUp()
    self.config = timesfm3_forecaster._ModelConfig(
      checkpoint_path="/fake/path",
      per_core_batch_size=2,
      input_patch_length=8,
      output_patch_length=8,
      median_quantile_index=1,
    )
    self._max_context = timesfm3_forecaster._MAX_CONTEXT_LENGTH
    timesfm3_forecaster._MAX_CONTEXT_LENGTH = 16

  def tearDown(self):
    timesfm3_forecaster._MAX_CONTEXT_LENGTH = self._max_context
    super().tearDown()

  def _forecaster(self):
    with mock.patch.object(
      timesfm3_forecaster.TimesFM3Forecaster, "_init_model", autospec=True
    ):
      forecaster = timesfm3_forecaster.TimesFM3Forecaster(self.config)
    forecaster.model = _RecordingFakeModel()
    forecaster.device = torch.device("cpu")
    return forecaster

  def test_truncated_context_keeps_covariate_window_aligned(self):
    # horizon=5 is not a multiple of output_patch_length=8, and the context
    # (40) exceeds the model context (16), so the query is truncated.
    forecaster = self._forecaster()
    context = np.arange(40, dtype=np.float32)
    pf_cov = np.arange(45, dtype=np.float32)  # context + horizon
    outputs = list(
      forecaster.predict_batch(
        contexts=[context], horizon=5, past_future_covariates=[pf_cov]
      )
    )
    call = forecaster.model.calls[-1]
    target = call["target"][0, 0]
    pf = call["past_future_covariates"][0, 0]
    # The covariate window must start at the same time step as the target
    # window and its future part must be exactly the requested horizon.
    self.assertEqual(float(target[0]), float(pf[0]))
    self.assertEqual(pf.shape[-1], target.shape[-1] + 5)
    self.assertEqual(call["horizon"], 5)
    self.assertEqual(outputs[0].forecast.shape, (5,))

  def test_mixed_length_batch_with_covariates(self):
    forecaster = self._forecaster()
    long_ctx = np.arange(40, dtype=np.float32)
    short_ctx = np.arange(100, 112, dtype=np.float32)
    outputs = list(
      forecaster.predict_batch(
        contexts=[long_ctx, short_ctx],
        horizon=5,
        past_future_covariates=[
          np.arange(45, dtype=np.float32),
          np.arange(100, 117, dtype=np.float32),
        ],
      )
    )
    self.assertEqual([o.forecast.shape for o in outputs], [(5,), (5,)])

  def test_predict_batch_accepts_2d_array(self):
    forecaster = self._forecaster()
    outputs = list(
      forecaster.predict_batch(
        contexts=np.random.rand(3, 12).astype(np.float32), horizon=4
      )
    )
    self.assertEqual(len(outputs), 3)


if __name__ == "__main__":
  unittest.main()
