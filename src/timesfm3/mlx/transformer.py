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

"""Transformer layers for the MLX TimesFM3 model.

Port of the PyTorch MixingTransformer: RoPE, per-head QK RMSNorm, Pax-style PerDimScale, and a
per-layer sequence-attention -> variate-attention -> FFN block.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from . import configs, normalization


def rope(
  x: mx.array, position: mx.array, min_ts: float = 1.0, max_ts: float = 10000.0
) -> mx.array:
  """Rotary positional embedding on ``(b, n, h, hd)`` with ``position`` ``(b, n)`` (half-split)."""
  hd = x.shape[-1]
  half = hd // 2
  fraction = 2.0 * mx.arange(half).astype(mx.float32) / hd
  timescale = min_ts * (max_ts / min_ts) ** fraction
  sinusoid = position[:, :, None, None].astype(mx.float32) / timescale.reshape(
    1, 1, 1, -1
  )
  sin, cos = mx.sin(sinusoid), mx.cos(sinusoid)
  first, second = mx.split(x, 2, axis=-1)
  return mx.concatenate(
    [first * cos - second * sin, second * cos + first * sin], axis=-1
  )


class MultiHeadAttention(nn.Module):
  """Multi-head attention with RoPE, per-head QK RMSNorm and PerDimScale.

  ``rescale_logits`` is False (memory-efficient-attention parity): the query is pre-multiplied by
  ``sqrt(head_dim)`` with no internal division, so the net logit scale is ``sqrt(head_dim)``.
  """

  def __init__(self, cfg: configs.TimesFM3MlxConfig, use_rope: bool, causal: bool):
    super().__init__()
    d = cfg.model_dims
    self.num_heads = cfg.num_heads
    self.head_dim = cfg.head_dim
    self.use_rope = use_rope
    self.causal = causal
    self.query_proj = nn.Linear(d, d, bias=False)
    self.key_proj = nn.Linear(d, d, bias=False)
    self.value_proj = nn.Linear(d, d, bias=False)
    self.out_proj = nn.Linear(d, d, bias=False)
    self.query_ln = normalization.RMSNorm(self.head_dim)
    self.key_ln = normalization.RMSNorm(self.head_dim)
    self.per_dim_scale = normalization.PerDimScale(self.head_dim)

  def __call__(self, x: mx.array, patch_mask: mx.array) -> mx.array:
    b, n, _ = x.shape
    # Single-position attention (variate attention over one variate): softmax over a single key is
    # exactly 1, so the output equals the value projection. Skip Q/K/RoPE/norms/softmax.
    if n == 1:
      return self.out_proj(self.value_proj(x))
    h, hd = self.num_heads, self.head_dim
    q = self.query_proj(x).reshape(b, n, h, hd)
    k = self.key_proj(x).reshape(b, n, h, hd)
    v = self.value_proj(x).reshape(b, n, h, hd)
    if self.use_rope:
      pos = mx.broadcast_to(mx.arange(n)[None, :], (b, n))
      q = rope(q, pos)
      k = rope(k, pos)
    q = self.query_ln(q)
    k = self.key_ln(k)
    q = self.per_dim_scale(q)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    kv_valid = (~patch_mask)[:, None, None, :]
    if self.causal:
      qi = mx.arange(n)[None, None, :, None]
      ki = mx.arange(n)[None, None, None, :]
      attend = (qi >= ki) & kv_valid
    else:
      attend = mx.broadcast_to(kv_valid, (b, 1, n, n))
    bias = mx.where(attend, 0.0, -1e9)
    q = q * math.sqrt(hd)
    logits = (q @ k.transpose(0, 1, 3, 2)) + bias
    w = mx.softmax(logits, axis=-1)
    out = w @ v
    out = out.transpose(0, 2, 1, 3).reshape(b, n, h * hd)
    return self.out_proj(out)


class MixingTransformer(nn.Module):
  """One layer: sequence attention, then variate attention, then FFN (each pre/post-RMSNorm'd)."""

  def __init__(self, cfg: configs.TimesFM3MlxConfig):
    super().__init__()
    d = cfg.model_dims
    self.pre_seq_attn_ln = normalization.RMSNorm(d)
    self.post_seq_attn_ln = normalization.RMSNorm(d)
    self.seq_attn = MultiHeadAttention(cfg, use_rope=True, causal=True)
    self.use_var = cfg.use_variate_attention
    if self.use_var:
      self.pre_var_attn_ln = normalization.RMSNorm(d)
      self.post_var_attn_ln = normalization.RMSNorm(d)
      self.var_attn = MultiHeadAttention(cfg, use_rope=False, causal=False)
    self.pre_ff_ln = normalization.RMSNorm(d)
    self.post_ff_ln = normalization.RMSNorm(d)
    self.ff0 = nn.Linear(d, cfg.hidden_dims, bias=False)
    self.ff1 = nn.Linear(cfg.hidden_dims, d, bias=False)

  def __call__(self, x: mx.array, patch_mask: mx.array) -> mx.array:
    b, v, n, d = x.shape
    # sequence attention over n, batched across (b, v)
    sa_in = self.pre_seq_attn_ln(x).reshape(b * v, n, d)
    sa = self.seq_attn(sa_in, patch_mask.reshape(b * v, n)).reshape(b, v, n, d)
    h1 = self.post_seq_attn_ln(sa) + x
    # variate attention over v, batched across (b, n)
    if self.use_var:
      va_in = self.pre_var_attn_ln(h1).transpose(0, 2, 1, 3).reshape(b * n, v, d)
      va_mask = patch_mask.transpose(0, 2, 1).reshape(b * n, v)
      va = self.var_attn(va_in, va_mask).reshape(b, n, v, d).transpose(0, 2, 1, 3)
      h2 = self.post_var_attn_ln(va) + h1
    else:
      h2 = h1
    ff = self.ff1(nn.relu(self.ff0(self.pre_ff_ln(h2))))
    return self.post_ff_ln(ff) + h2


class StackedMixingTransformer(nn.Module):
  """A stack of ``num_layers`` MixingTransformer layers."""

  def __init__(self, cfg: configs.TimesFM3MlxConfig):
    super().__init__()
    self.layers = [MixingTransformer(cfg) for _ in range(cfg.num_layers)]

  def __call__(self, x: mx.array, patch_mask: mx.array) -> mx.array:
    for layer in self.layers:
      x = layer(x, patch_mask)
    return x
