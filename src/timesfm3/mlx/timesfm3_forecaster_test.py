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

"""Tests for the MLX TimesFM3 backend.

The lightweight tests use a tiny randomly-initialized model (no download). The real-weight tests
are skipped unless the ``google/timesfm-3.0-pytorch`` checkpoint is cached, and the parity test is
additionally skipped unless PyTorch is available to serve as the oracle. MLX runs on Apple silicon,
so the whole module is skipped when MLX is not installed.
"""

import os
import unittest

import numpy as np

try:
  import mlx.core as mx

  from . import configs
  from . import model as mlx_model_lib
  from . import timesfm3_forecaster

  _HAS_MLX = True
except ImportError:
  _HAS_MLX = False

_CHECKPOINT = "google/timesfm-3.0-pytorch"


def _weights_cached() -> bool:
  try:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(_CHECKPOINT, "model.safetensors")
    return isinstance(path, str) and os.path.exists(path)
  except Exception:
    return False


def _torch_available() -> bool:
  try:
    import torch  # noqa: F401

    return True
  except ImportError:
    return False


@unittest.skipUnless(_HAS_MLX, "mlx is not installed (Apple silicon only)")
class TimesFM3MlxModelTest(unittest.TestCase):
  """Structural tests that need no pretrained weights."""

  def _tiny_model(self):
    cfg = configs.TimesFM3MlxConfig(model_dims=64, num_layers=2, num_heads=4)
    return mlx_model_lib.TimesFM3Mlx(cfg, compile=False)

  def test_decode_shape(self):
    model = self._tiny_model()
    ctx = mx.array(np.sin(np.linspace(0, 10, 128)).astype(np.float32))[None, None, :]
    logits = model.decode(ctx, horizon=24)
    self.assertEqual(logits.shape[:3], (1, 1, 24))
    self.assertEqual(logits.shape[3], 9)  # 9 deciles

  def test_batched_decode_matches_single(self):
    # decode() is per-series independent, so a batched call must equal looping series-by-series.
    model = self._tiny_model()
    ctxs = np.stack([np.sin(np.linspace(0, 10 + i, 128)) for i in range(3)]).astype(
      np.float32
    )
    batched = np.array(model.decode(mx.array(ctxs)[:, None, :], horizon=24))
    for i in range(3):
      single = np.array(model.decode(mx.array(ctxs[i])[None, None, :], horizon=24))
      np.testing.assert_allclose(batched[i], single[0], atol=1e-4)

  def test_decode_multivariate_targets_shape(self):
    # Two target variates in, two forecasts out.
    model = self._tiny_model()
    ctx = mx.array(np.random.RandomState(0).randn(1, 2, 128).astype(np.float32))
    logits = model.decode(ctx, horizon=24)
    self.assertEqual(logits.shape, (1, 2, 24, 9))

  def test_decode_with_covariates_returns_all_variates(self):
    # decode() stacks targets, past-only and past-future covariates on the
    # variate axis, mirroring the torch backend, and returns one row per variate.
    model = self._tiny_model()
    rng = np.random.RandomState(0)
    target = mx.array(rng.randn(1, 2, 128).astype(np.float32))
    po = mx.array(rng.randn(1, 1, 128).astype(np.float32))
    pf = mx.array(rng.randn(1, 1, 128 + 24).astype(np.float32))
    logits = model.decode(
      target, horizon=24, past_only_covariates=po, past_future_covariates=pf
    )
    # 2 targets + 1 past-only + 1 past-future = 4 variates.
    self.assertEqual(logits.shape, (1, 4, 24, 9))

  def test_past_future_covariate_infers_horizon(self):
    # When past-future covariates are given, the horizon is read from their width.
    model = self._tiny_model()
    rng = np.random.RandomState(1)
    target = mx.array(rng.randn(1, 1, 128).astype(np.float32))
    pf = mx.array(rng.randn(1, 1, 128 + 30).astype(np.float32))
    logits = model.decode(target, past_future_covariates=pf)
    self.assertEqual(logits.shape[2], 30)

  def test_from_hf_config_rejects_frozen_running_stats(self):
    # The MLX backend does not implement frozen running stats. A checkpoint that
    # asks for them must fail loudly rather than silently diverge from torch.
    with self.assertRaises(NotImplementedError):
      configs.TimesFM3MlxConfig.from_hf_config({"use_frozen_running_stats": True})
    # The default (absent / False) builds normally.
    configs.TimesFM3MlxConfig.from_hf_config({"use_frozen_running_stats": False})

  def test_detrend_activates_on_strong_trend_only(self):
    # Detrend is data-driven (weight-independent): a near-linear series activates
    # it; a stationary oscillation does not.
    model = self._tiny_model()
    masks = mx.zeros((1, 1, 128), dtype=mx.bool_)
    trend = (np.linspace(0, 10, 128) + 0.05 * np.sin(np.linspace(0, 20, 128))).astype(
      np.float32
    )
    flat = np.sin(np.linspace(0, 20, 128)).astype(np.float32)
    self.assertTrue(
      bool(model._detrend(mx.array(trend)[None, None], masks, 128)[2][0, 0])
    )
    self.assertFalse(
      bool(model._detrend(mx.array(flat)[None, None], masks, 128)[2][0, 0])
    )


@unittest.skipUnless(
  _HAS_MLX and _weights_cached(),
  "requires the cached google/timesfm-3.0-pytorch checkpoint",
)
class TimesFM3MlxRealWeightsTest(unittest.TestCase):
  """End-to-end tests on the real pretrained weights."""

  def test_predict_shapes(self):
    forecaster = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(_CHECKPOINT)
    out = forecaster.predict(
      np.sin(np.linspace(0, 40, 512)).astype(np.float32),
      horizon=64,
      return_quantiles=True,
    )
    self.assertEqual(np.asarray(out.forecast).shape, (64,))
    self.assertEqual(np.asarray(out.quantiles).shape, (64, 9))

  def test_global_context_default_matches_torch(self):
    # The MLX backend must expose the same 15,360-step context cap as the torch
    # backend so the two are backend-equivalent for very long contexts.
    forecaster = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(_CHECKPOINT)
    self.assertEqual(forecaster.global_context, 15360)

  def test_context_truncated_to_global_context(self):
    # A context longer than the cap must be truncated to its last
    # `global_context` points before decode, so forecasting the full series
    # equals forecasting only that tail.
    cap = 256
    forecaster = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(
      _CHECKPOINT, max_context_length=cap
    )
    self.assertEqual(forecaster.global_context, cap)
    ctx = np.sin(np.linspace(0, 60, 512)).astype(np.float32)
    out_full = forecaster.predict(ctx, horizon=32, return_quantiles=True)
    out_tail = forecaster.predict(ctx[-cap:], horizon=32, return_quantiles=True)
    np.testing.assert_allclose(out_full.forecast, out_tail.forecast, atol=1e-5)
    np.testing.assert_allclose(out_full.quantiles, out_tail.quantiles, atol=1e-5)

  @unittest.skipUnless(_torch_available(), "requires torch as the parity oracle")
  def test_parity_with_torch_backend(self):
    from ..torch import timesfm3_forecaster as torch_forecaster

    ctx = np.sin(np.linspace(0, 40, 512)).astype(np.float32)
    mlx_out = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(
      _CHECKPOINT
    ).predict(ctx, horizon=64, return_quantiles=True)
    torch_out = torch_forecaster.TimesFM3Forecaster.from_pretrained(
      _CHECKPOINT
    ).predict(ctx, horizon=64, return_quantiles=True)
    self.assertLess(
      np.abs(np.asarray(mlx_out.forecast) - np.asarray(torch_out.forecast)).max(), 1e-3
    )
    self.assertLess(
      np.abs(np.asarray(mlx_out.quantiles) - np.asarray(torch_out.quantiles)).max(),
      1e-3,
    )


@unittest.skipUnless(
  _HAS_MLX and _weights_cached() and _torch_available(),
  "requires the cached checkpoint and torch as the parity oracle",
)
class TimesFM3MlxCovariateParityTest(unittest.TestCase):
  """Parity of the multivariate / covariate / long-horizon paths against torch."""

  @classmethod
  def setUpClass(cls):
    from ..torch import timesfm3_forecaster as torch_forecaster

    cls.mlx = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(_CHECKPOINT)
    cls.torch = torch_forecaster.TimesFM3Forecaster.from_pretrained(_CHECKPOINT)

  def _assert_parity(self, mlx_out, torch_out, atol=1e-3):
    np.testing.assert_allclose(
      np.asarray(mlx_out.forecast), np.asarray(torch_out.forecast), atol=atol
    )
    np.testing.assert_allclose(
      np.asarray(mlx_out.quantiles), np.asarray(torch_out.quantiles), atol=atol
    )

  def test_parity_long_horizon(self):
    # Horizon >= 128 exercises multiple output patches + CPM refine + stitching.
    ctx = np.sin(np.linspace(0, 40, 512)).astype(np.float32)
    for horizon in (128, 256):
      mlx_out = self.mlx.predict(ctx, horizon=horizon, return_quantiles=True)
      torch_out = self.torch.predict(ctx, horizon=horizon, return_quantiles=True)
      self.assertEqual(np.asarray(mlx_out.forecast).shape, (horizon,))
      self._assert_parity(mlx_out, torch_out)

  def test_parity_two_targets(self):
    rng = np.random.RandomState(0)
    target = np.stack(
      [
        np.sin(np.linspace(0, 30, 256)),
        np.cos(np.linspace(0, 18, 256)) + 0.1 * rng.randn(256),
      ]
    ).astype(np.float32)
    mlx_out = self.mlx.predict(target, horizon=64, return_quantiles=True)
    torch_out = self.torch.predict(target, horizon=64, return_quantiles=True)
    self.assertEqual(np.asarray(mlx_out.forecast).shape, (2, 64))
    self._assert_parity(mlx_out, torch_out)

  def test_parity_two_targets_with_covariates(self):
    rng = np.random.RandomState(1)
    ctx_len, horizon = 256, 32
    target = np.stack(
      [np.sin(np.linspace(0, 24, ctx_len)), np.sin(np.linspace(1, 26, ctx_len))]
    ).astype(np.float32)
    past_only = (0.5 * rng.randn(1, ctx_len)).astype(np.float32)
    past_future = np.sin(np.linspace(0, 30, ctx_len + horizon))[None, :].astype(
      np.float32
    )
    kw = dict(
      horizon=horizon,
      past_only_covariates=past_only,
      past_future_covariates=past_future,
      return_quantiles=True,
    )
    mlx_out = self.mlx.predict(target, **kw)
    torch_out = self.torch.predict(target, **kw)
    self.assertEqual(np.asarray(mlx_out.forecast).shape, (2, horizon))
    self._assert_parity(mlx_out, torch_out)

  def test_parity_detrend_activated(self):
    # A strong linear trend activates detrend on both backends; outputs must agree.
    ctx = (np.linspace(0, 12, 512) + 0.05 * np.sin(np.linspace(0, 40, 512))).astype(
      np.float32
    )
    masks = mx.zeros((1, 1, 512), dtype=mx.bool_)
    self.assertTrue(
      bool(self.mlx.model._detrend(mx.array(ctx)[None, None], masks, 512)[2][0, 0])
    )
    mlx_out = self.mlx.predict(ctx, horizon=64, return_quantiles=True)
    torch_out = self.torch.predict(ctx, horizon=64, return_quantiles=True)
    self._assert_parity(mlx_out, torch_out)


if __name__ == "__main__":
  unittest.main()
