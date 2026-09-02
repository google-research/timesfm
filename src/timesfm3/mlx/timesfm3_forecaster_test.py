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
    ctxs = np.stack(
      [np.sin(np.linspace(0, 10 + i, 128)) for i in range(3)]
    ).astype(np.float32)
    batched = np.array(model.decode(mx.array(ctxs)[:, None, :], horizon=24))
    for i in range(3):
      single = np.array(model.decode(mx.array(ctxs[i])[None, None, :], horizon=24))
      np.testing.assert_allclose(batched[i], single[0], atol=1e-4)


@unittest.skipUnless(
  _HAS_MLX and _weights_cached(), "requires the cached google/timesfm-3.0-pytorch checkpoint"
)
class TimesFM3MlxRealWeightsTest(unittest.TestCase):
  """End-to-end tests on the real pretrained weights."""

  def test_predict_shapes(self):
    forecaster = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(_CHECKPOINT)
    out = forecaster.predict(
      np.sin(np.linspace(0, 40, 512)).astype(np.float32), horizon=64, return_quantiles=True
    )
    self.assertEqual(np.asarray(out.forecast).shape, (64,))
    self.assertEqual(np.asarray(out.quantiles).shape, (64, 9))

  @unittest.skipUnless(_torch_available(), "requires torch as the parity oracle")
  def test_parity_with_torch_backend(self):
    from ..torch import timesfm3_forecaster as torch_forecaster

    ctx = np.sin(np.linspace(0, 40, 512)).astype(np.float32)
    mlx_out = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(_CHECKPOINT).predict(
      ctx, horizon=64, return_quantiles=True
    )
    torch_out = torch_forecaster.TimesFM3Forecaster.from_pretrained(_CHECKPOINT).predict(
      ctx, horizon=64, return_quantiles=True
    )
    self.assertLess(
      np.abs(np.asarray(mlx_out.forecast) - np.asarray(torch_out.forecast)).max(), 1e-3
    )
    self.assertLess(
      np.abs(np.asarray(mlx_out.quantiles) - np.asarray(torch_out.quantiles)).max(), 1e-3
    )


if __name__ == "__main__":
  unittest.main()
