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

"""MLX-native TimesFM3 model (inference only).

A faithful translation of the PyTorch ``TimesFM3Torch`` model to Apple MLX. Module attribute names
mirror the checkpoint tensor names exactly, so loading a ``model.safetensors`` is a near-identity
name map. Numerically verified against the PyTorch backend to ~1e-6 across horizons.
"""

from __future__ import annotations

import json
import math
import os

import mlx.core as mx
import mlx.nn as nn

from . import cpm_revin_refine as cpm_revin_refine_lib
from . import configs, transformer, util
from .dense import ResidualBlock


class TimesFM3Mlx(nn.Module):
  """MLX TimesFM3. Use :meth:`from_pretrained` to load real weights from a checkpoint."""

  def __init__(
    self, config: configs.TimesFM3MlxConfig | None = None, compile: bool = True
  ):
    super().__init__()
    cfg = config or configs.TimesFM3MlxConfig()
    self.config = cfg
    self.pre_transformer_resblock = ResidualBlock(
      2 * (cfg.input_patch_len + cfg.output_patch_len), cfg.model_dims
    )
    self.transformer_stack = transformer.StackedMixingTransformer(cfg)
    self.output_head = nn.Linear(
      cfg.model_dims, cfg.output_patch_len * cfg.num_quantiles, bias=True
    )
    self._compute_dtype = mx.float32
    self._use_compile = compile
    self._compiled_forward = None

  # ---- full-sequence forward over patched inputs (target-only path) ----
  def _forward_logits(self, values, masks, patch_is_target, patch_cpm_mask=None):
    cfg = self.config
    running_n, mu, sigma = util.get_running_stats(values, masks)
    vals_norm = util.revin(values, mu, sigma)
    vals_norm = mx.where(masks, 0.0, vals_norm)
    vals_fcov, wrap = util.output_patch_via_roll(values, cfg.rolls)
    vals_fcov = util.revin(vals_fcov, mu, sigma)
    masks_fcov_raw, _ = util.output_patch_via_roll(masks.astype(mx.float32), cfg.rolls)
    masks_fcov = (masks_fcov_raw > 0.5) | patch_is_target[..., None] | wrap
    vals_fcov = mx.where(masks_fcov, 0.0, vals_fcov)
    vals_cat = mx.concatenate([vals_norm, vals_fcov], axis=-1)
    masks_cat = mx.concatenate([masks, masks_fcov], axis=-1)
    resblock_in = mx.concatenate([vals_cat, masks_cat.astype(mx.float32)], axis=-1)
    x = self.pre_transformer_resblock(resblock_in.astype(self._compute_dtype))
    patch_mask = masks_cat.astype(mx.float32).min(axis=3) > 0.5
    eff = (
      mx.cumprod(patch_mask.astype(mx.int32), axis=2) > 0
    )  # mask leading patches only
    x = self.transformer_stack(x, eff)
    raw = self.output_head(x).astype(mx.float32)
    if patch_cpm_mask is not None:
      ref_mu, ref_sigma = cpm_revin_refine_lib.cpm_iterative_revin_refine(
        raw,
        running_n,
        mu,
        sigma,
        patch_cpm_mask,
        cfg.num_quantiles // 2,
        cfg.rolls,
        cfg.input_patch_len,
        cfg.num_quantiles,
        cfg.value_clip,
      )
      cpm = patch_cpm_mask[:, None, :]
      mu = mx.where(cpm, ref_mu, mu)
      sigma = mx.where(cpm, ref_sigma, sigma)
    raw = util.revin(raw, mu, sigma, reverse=True)
    raw = mx.clip(raw, -cfg.value_clip, cfg.value_clip)
    b, v, n = raw.shape[:3]
    return raw.reshape(b, v, n, cfg.output_patch_len, cfg.num_quantiles)

  def _forward_fn(self):
    """Return the forward, ``mx.compile``d once (lazily, after weights are loaded).

    Compiling fuses the whole pass (including the unrolled running-stats / CPM-refine loops) into
    one graph, removing the per-op and Python-loop dispatch overhead that dominates latency at this
    model size. MLX recompiles per unique input shape and caches.
    """
    if not self._use_compile:
      return self._forward_logits
    if self._compiled_forward is None:
      self._compiled_forward = mx.compile(self._forward_logits)
    return self._compiled_forward

  def decode(self, target: mx.array, horizon: int) -> mx.array:
    """``target`` ``(b, 1, context)`` -> logits ``(b, 1, horizon, num_quantiles)`` (target-only)."""
    cfg = self.config
    b, num_target, context = target.shape
    p = cfg.input_patch_len
    pad = (p - (context % p)) % p
    mask = mx.zeros((b, context + pad), dtype=mx.bool_)
    if pad:
      target = mx.concatenate([mx.zeros((b, num_target, pad)), target], axis=-1)
      mask = mx.concatenate(
        [
          mx.ones((b, pad), dtype=mx.bool_),
          mx.zeros((b, context), dtype=mx.bool_),
        ],
        axis=-1,
      )
      context = context + pad
    num_ctx_patches = context // p

    extract_len = min(2 * p, cfg.output_patch_len)
    overlap = extract_len - p
    num_forecast_patches = max(math.ceil((horizon - overlap) / p), 1)
    num_hor_patches = num_forecast_patches + cfg.rolls - 1
    padded_h = num_hor_patches * p

    ctx_masks = mx.broadcast_to(mask[:, None, :], (b, num_target, context))
    ctx_vals = target

    m_trend, c_trend, apply_detrend = self._detrend(ctx_vals, ctx_masks, context)
    if cfg.use_linear_detrending:
      t = mx.arange(-(context - 1), 1).astype(mx.float32)[None, None, :] / context
      detr = ctx_vals - (m_trend[..., None] * t + c_trend[..., None])
      ctx_vals = mx.where(apply_detrend[..., None], detr, ctx_vals)
    ctx_vals = mx.where(ctx_masks, 0.0, ctx_vals)

    hor_vals = mx.zeros((b, num_target, padded_h))
    hor_masks = mx.ones((b, num_target, padded_h), dtype=mx.bool_)
    all_vals = mx.concatenate([ctx_vals, hor_vals], axis=-1)
    all_masks = mx.concatenate([ctx_masks, hor_masks], axis=-1)

    n_tot = num_ctx_patches + num_hor_patches
    values_bvnp = all_vals.reshape(b, num_target, n_tot, p)
    masks_bvnp = all_masks.reshape(b, num_target, n_tot, p)
    patch_is_target = mx.ones((b, num_target, n_tot), dtype=mx.bool_)

    horizon_cpm = mx.concatenate(
      [
        mx.zeros((b, num_ctx_patches), dtype=mx.bool_),
        mx.ones((b, num_hor_patches), dtype=mx.bool_),
      ],
      axis=1,
    )
    logits = self._forward_fn()(values_bvnp, masks_bvnp, patch_is_target, horizon_cpm)

    fidx = mx.arange(num_forecast_patches) + (num_ctx_patches - 1)
    patch_preds = mx.take(logits, fidx, axis=2)[:, :, :, :extract_len, :]
    horizon_logits = util.stitch_patches(patch_preds, p)[:, :, :horizon, :]

    if cfg.use_linear_detrending:
      tf = mx.arange(1, horizon + 1).astype(mx.float32) / context
      trend = m_trend[:, :, None] * tf[None, None, :] + c_trend[:, :, None]
      trend = mx.where(apply_detrend[:, :, None], trend, 0.0)
      horizon_logits = horizon_logits + trend[:, :, :, None]
    return horizon_logits

  def _detrend(self, ctx_vals, ctx_masks, context):
    cfg = self.config
    b, v, _ = ctx_vals.shape
    if not cfg.use_linear_detrending:
      z = mx.zeros((b, v))
      return z, z, mx.zeros((b, v), dtype=mx.bool_)
    t = mx.arange(-(context - 1), 1).astype(mx.float32)[None, None, :] / context
    n_v = (~ctx_masks).astype(mx.float32).sum(axis=-1)
    sum_t = mx.where(~ctx_masks, t, 0.0).sum(axis=-1)
    sum_t2 = mx.where(~ctx_masks, t * t, 0.0).sum(axis=-1)
    sum_y = mx.where(~ctx_masks, ctx_vals, 0.0).sum(axis=-1)
    sum_ty = mx.where(~ctx_masks, t * ctx_vals, 0.0).sum(axis=-1)
    det = n_v * sum_t2 - sum_t**2
    safe = mx.where(det == 0.0, 1.0, det)
    m = mx.where(det == 0.0, 0.0, (n_v * sum_ty - sum_t * sum_y) / safe)
    c = mx.where(
      det == 0.0,
      mx.where(n_v > 0, sum_y / mx.maximum(n_v, 1.0), 0.0),
      (sum_y - m * sum_t) / mx.maximum(n_v, 1.0),
    )
    detr = ctx_vals - (m[..., None] * t + c[..., None])
    mean_y = sum_y / mx.maximum(n_v, 1.0)
    sum_y2 = mx.where(~ctx_masks, ctx_vals**2, 0.0).sum(axis=-1)
    std_orig = mx.sqrt(mx.maximum(sum_y2 / mx.maximum(n_v, 1.0) - mean_y**2, 0.0))
    sum_yd = mx.where(~ctx_masks, detr, 0.0).sum(axis=-1)
    mean_yd = sum_yd / mx.maximum(n_v, 1.0)
    sum_yd2 = mx.where(~ctx_masks, detr**2, 0.0).sum(axis=-1)
    std_det = mx.sqrt(mx.maximum(sum_yd2 / mx.maximum(n_v, 1.0) - mean_yd**2, 0.0))
    apply = std_det < cfg.linear_detrending_threshold * std_orig
    return m, c, apply

  # ---- loading ----
  def load_safetensors(self, path: str) -> "TimesFM3Mlx":
    """Load a ``model.safetensors`` checkpoint (near-identity name map) into this model."""
    from mlx.utils import tree_flatten, tree_unflatten
    from safetensors.numpy import load_file

    state = load_file(path)
    ckpt = {k: mx.array(v) for k, v in state.items()}
    model_keys = {k for k, _ in tree_flatten(self.parameters())}
    missing = model_keys - set(ckpt)
    extra = set(ckpt) - model_keys
    if missing or extra:
      raise ValueError(
        f"weight/name mismatch: {len(missing)} model params unmapped "
        f"(e.g. {sorted(missing)[:3]}), {len(extra)} checkpoint tensors unused "
        f"(e.g. {sorted(extra)[:3]})"
      )
    self.update(tree_unflatten(list(ckpt.items())))
    mx.eval(self.parameters())
    return self

  @classmethod
  def from_pretrained(
    cls,
    checkpoint_path: str = "google/timesfm-3.0-pytorch",
    *,
    compile: bool = True,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
  ) -> "TimesFM3Mlx":
    """Load config + weights from a local directory or a Hugging Face repo id."""
    checkpoint_path = os.path.expanduser(checkpoint_path)
    if os.path.isdir(checkpoint_path):
      config_file = os.path.join(checkpoint_path, "config.json")
      weights_file = os.path.join(checkpoint_path, "model.safetensors")
    else:
      from huggingface_hub import hf_hub_download

      dl = dict(
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
        force_download=force_download,
      )
      config_file = hf_hub_download(checkpoint_path, "config.json", **dl)
      weights_file = hf_hub_download(checkpoint_path, "model.safetensors", **dl)
    with open(config_file) as f:
      hf_cfg = json.load(f)
    model = cls(configs.TimesFM3MlxConfig.from_hf_config(hf_cfg), compile=compile)
    model.load_safetensors(weights_file)
    return model
