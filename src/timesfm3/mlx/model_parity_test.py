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

"""Numerical parity tests between the MLX and PyTorch TimesFM3 backends.

These tests build tiny (model_dims=32, 2-layer) randomly-initialized models on CPU -- no
pretrained checkpoint download and no GPU/Metal requirement -- transplant the PyTorch model's
weights into the MLX model via a temporary safetensors file (the two backends use an
identical parameter-name/shape scheme by design, see ``TimesFM3Mlx.load_safetensors``), and
compare ``decode()`` outputs directly.

``TimesFM3ParityTest`` checks configurations both backends are meant to support identically.
``TimesFM3KnownDivergenceTest`` is a regression suite for specific MLX/PyTorch divergences found
during review (``use_stitching=False``, ``use_iterative_cpm_revin=False``, residual block
``activation``, and unmasked NaN in the context) and since fixed in the MLX backend.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

try:
  import mlx.core as mx
  from mlx.utils import tree_flatten

  from . import configs as mlx_configs
  from . import model as mlx_model_lib

  _HAS_MLX = True
except ImportError:
  _HAS_MLX = False


def _torch_available() -> bool:
  try:
    import torch  # noqa: F401

    return True
  except ImportError:
    return False


_INPUT_PATCH_LEN = 8
_OUTPUT_PATCH_LEN = 16
_MODEL_DIMS = 32
_HIDDEN_DIMS = 32
_NUM_HEADS = 4
_NUM_LAYERS = 2
_QUANTILES = [0.1, 0.5, 0.9]


def _build_torch_model(
  *,
  use_stitching: bool = True,
  use_iterative_cpm_revin: bool = True,
  activation: str = "relu",
  ff_activation: str = "relu",
  v_norm: str = "none",
  causal_attention: bool = True,
  use_rope_var: bool = False,
  use_memory_efficient_attention: bool = True,
  seed: int = 0,
):
  import torch

  from ..torch import configs as torch_configs
  from ..torch import model as torch_model_lib

  torch.manual_seed(seed)
  resblock_cfg = torch_configs.ResidualBlockConfig(
    hidden_dims=_MODEL_DIMS,
    output_dims=_MODEL_DIMS,
    use_bias=False,
    activation=activation,
  )
  transformer_cfg = torch_configs.StackedTransformersConfig(
    num_layers=_NUM_LAYERS,
    use_remat=False,
    transformer=torch_configs.TransformerConfig(
      model_dims=_MODEL_DIMS,
      hidden_dims=_HIDDEN_DIMS,
      num_heads=_NUM_HEADS,
      attention_norm="rms",
      feedforward_norm="rms",
      qk_norm="rms",
      v_norm=v_norm,
      use_rope_seq=True,
      use_rope_var=use_rope_var,
      use_bias=False,
      ff_activation=ff_activation,
      deterministic=True,
      use_sdpa=True,
      causal_attention=causal_attention,
      use_memory_efficient_attention=use_memory_efficient_attention,
    ),
  )
  model = torch_model_lib.TimesFM3Torch(
    input_patch_len=_INPUT_PATCH_LEN,
    output_patch_len=_OUTPUT_PATCH_LEN,
    quantiles=_QUANTILES,
    residual_block_config=resblock_cfg,
    transformer_config=transformer_cfg,
    use_stitching=use_stitching,
    use_linear_detrending=True,
    use_iterative_cpm_revin=use_iterative_cpm_revin,
    use_frozen_running_stats=False,
  )
  model.eval()
  return model


def _build_mlx_model(
  *,
  use_stitching: bool = True,
  use_iterative_cpm_revin: bool = True,
  residual_activation: str = "relu",
  ff_activation: str = "relu",
  v_norm: str = "none",
  causal_attention: bool = True,
  use_rope_var: bool = False,
  use_memory_efficient_attention: bool = True,
) -> "mlx_model_lib.TimesFM3Mlx":
  cfg = mlx_configs.TimesFM3MlxConfig(
    input_patch_len=_INPUT_PATCH_LEN,
    output_patch_len=_OUTPUT_PATCH_LEN,
    model_dims=_MODEL_DIMS,
    hidden_dims=_HIDDEN_DIMS,
    num_layers=_NUM_LAYERS,
    num_heads=_NUM_HEADS,
    quantiles=tuple(_QUANTILES),
    use_variate_attention=True,
    use_stitching=use_stitching,
    use_linear_detrending=True,
    linear_detrending_threshold=0.5,
    value_clip=1e20,
    use_iterative_cpm_revin=use_iterative_cpm_revin,
    residual_activation=residual_activation,
    ff_activation=ff_activation,
    v_norm=v_norm,
    causal_attention=causal_attention,
    use_rope_var=use_rope_var,
    use_memory_efficient_attention=use_memory_efficient_attention,
  )
  return mlx_model_lib.TimesFM3Mlx(cfg, compile=False)


def _transplant(torch_model, mlx_model) -> "mlx_model_lib.TimesFM3Mlx":
  """Copies ``torch_model``'s weights into ``mlx_model`` via a safetensors round-trip."""
  from safetensors.torch import save_file

  state_dict = torch_model.state_dict()
  with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "model.safetensors")
    save_file({k: v.contiguous() for k, v in state_dict.items()}, path)
    mlx_model.load_safetensors(path)
  return mlx_model


def _build_pair(**torch_kwargs):
  """Builds a (torch_model, mlx_model) pair sharing identical weights."""
  torch_model = _build_torch_model(**torch_kwargs)
  mlx_model = _build_mlx_model(
    use_stitching=torch_kwargs.get("use_stitching", True),
    use_iterative_cpm_revin=torch_kwargs.get("use_iterative_cpm_revin", True),
    residual_activation=torch_kwargs.get("activation", "relu"),
    ff_activation=torch_kwargs.get("ff_activation", "relu"),
    v_norm=torch_kwargs.get("v_norm", "none"),
    causal_attention=torch_kwargs.get("causal_attention", True),
    use_rope_var=torch_kwargs.get("use_rope_var", False),
    use_memory_efficient_attention=torch_kwargs.get("use_memory_efficient_attention", True),
  )
  _transplant(torch_model, mlx_model)
  return torch_model, mlx_model


def _decode_both(torch_model, mlx_model, ctx: np.ndarray, horizon: int, **kwargs):
  import torch

  torch_kwargs = {k: torch.from_numpy(v) for k, v in kwargs.items()}
  mlx_kwargs = {k: mx.array(v) for k, v in kwargs.items()}
  with torch.no_grad():
    torch_out = torch_model.decode(
      target=torch.from_numpy(ctx), horizon=horizon, **torch_kwargs
    ).numpy()
  mlx_out = np.array(mlx_model.decode(mx.array(ctx), horizon=horizon, **mlx_kwargs))
  return torch_out, mlx_out


@unittest.skipUnless(_HAS_MLX, "mlx is not installed (Apple silicon only)")
@unittest.skipUnless(_torch_available(), "requires torch as the parity oracle")
class TimesFM3ParityTest(unittest.TestCase):
  """Configurations both backends claim to support identically -- must match closely."""

  def test_decode_univariate_matches(self):
    torch_model, mlx_model = _build_pair()
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_decode_multivariate_with_covariates_matches(self):
    torch_model, mlx_model = _build_pair()
    rng = np.random.RandomState(1)
    target = rng.randn(1, 2, 128).astype(np.float32)
    po = rng.randn(1, 1, 128).astype(np.float32)
    pf = rng.randn(1, 1, 128 + 24).astype(np.float32)
    torch_out, mlx_out = _decode_both(
      torch_model,
      mlx_model,
      target,
      horizon=24,
      past_only_covariates=po,
      past_future_covariates=pf,
    )
    self.assertEqual(torch_out.shape, (1, 4, 24, 3))
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)


@unittest.skipUnless(_HAS_MLX, "mlx is not installed (Apple silicon only)")
@unittest.skipUnless(_torch_available(), "requires torch as the parity oracle")
class TimesFM3KnownDivergenceTest(unittest.TestCase):
  """Regression tests for specific MLX/PyTorch numerical divergences found during review, now
  fixed in the MLX backend.
  """

  def test_use_stitching_false_is_ignored_by_mlx(self):
    # mlx/model.py's decode() branches on cfg.use_stitching the same way torch/model.py does.
    torch_model, mlx_model = _build_pair(use_stitching=False)
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_use_iterative_cpm_revin_false_is_ignored_by_mlx(self):
    # torch/model.py only calls cpm_iterative_revin_refine when use_iterative_cpm_revin is True;
    # mlx/model.py mirrors this via TimesFM3MlxConfig.use_iterative_cpm_revin.
    torch_model, mlx_model = _build_pair(use_iterative_cpm_revin=False)
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_use_iterative_cpm_revin_long_horizon(self):
    # horizon=24 above only exercises one block_offset wraparound (rolls=2). A longer horizon
    # forces cpm_iterative_revin_refine through multiple wraps, accumulating anchor predictions
    # across consecutive rolled steps.
    for use_cpm in (True, False):
      torch_model, mlx_model = _build_pair(use_iterative_cpm_revin=use_cpm)
      ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
      torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=48)
      np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_use_iterative_cpm_revin_with_covariates(self):
    # The refine step operates on (b, v, rolls, patch_len) where v mixes targets and covariates;
    # make sure it agrees with torch when past_only/past_future covariates are present too.
    rng = np.random.RandomState(42)
    target = rng.randn(1, 2, 128).astype(np.float32)
    po = rng.randn(1, 1, 128).astype(np.float32)
    pf = rng.randn(1, 1, 128 + 32).astype(np.float32)
    for use_cpm in (True, False):
      torch_model, mlx_model = _build_pair(use_iterative_cpm_revin=use_cpm)
      torch_out, mlx_out = _decode_both(
        torch_model,
        mlx_model,
        target,
        horizon=32,
        past_only_covariates=po,
        past_future_covariates=pf,
      )
      np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_residual_block_activation_is_ignored_by_mlx(self):
    # mlx/dense.py's ResidualBlock reads its activation from
    # TimesFM3MlxConfig.residual_activation, matching torch's ResidualBlockConfig.activation.
    torch_model, mlx_model = _build_pair(activation="swish")
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_nan_context_is_not_sanitized_by_mlx(self):
    # torch/model.py's forward() runs `values = torch.nan_to_num(values, nan=0.0)` (then clamps)
    # before computing RevIN running stats; mlx/model.py's _forward_logits mirrors this.
    torch_model, mlx_model = _build_pair()
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    ctx[0, 0, 50] = np.nan
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    self.assertFalse(np.isnan(mlx_out).any(), "mlx output contains NaN where torch's does not")
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_ff_activation_is_ignored_by_mlx(self):
    # mlx/transformer.py's MixingTransformer reads its FFN activation from
    # TimesFM3MlxConfig.ff_activation, matching torch's TransformerConfig.ff_activation.
    torch_model, mlx_model = _build_pair(ff_activation="swish")
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_v_norm_is_not_implemented_by_mlx(self):
    # mlx/transformer.py's MultiHeadAttention applies value normalization when
    # cfg.v_norm == "rms", including through the single-variate (n==1) attention shortcut.
    torch_model, mlx_model = _build_pair(v_norm="rms")
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_causal_attention_false_is_ignored_by_mlx(self):
    # mlx/transformer.py's sequence attention reads causality from cfg.causal_attention instead
    # of always being causal.
    torch_model, mlx_model = _build_pair(causal_attention=False)
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_use_memory_efficient_attention_false_is_ignored_by_mlx(self):
    # mlx/transformer.py's attention logit scale follows cfg.use_memory_efficient_attention
    # instead of always assuming it's True.
    torch_model, mlx_model = _build_pair(use_memory_efficient_attention=False)
    ctx = np.random.RandomState(0).randn(1, 1, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)

  def test_use_rope_var_true_is_ignored_by_mlx(self):
    # mlx/transformer.py's variate attention reads its RoPE flag from cfg.use_rope_var. A
    # 1-variate context can't detect this: softmax over a single key is always 1.0 regardless of
    # RoPE, so this needs >=2 variates to actually exercise variate attention.
    torch_model, mlx_model = _build_pair(use_rope_var=True)
    ctx = np.random.RandomState(2).randn(1, 2, 128).astype(np.float32)
    torch_out, mlx_out = _decode_both(torch_model, mlx_model, ctx, horizon=24)
    np.testing.assert_allclose(torch_out, mlx_out, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
