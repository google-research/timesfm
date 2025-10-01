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

"""Transformer layers for TimesFM."""

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .. import configs
from . import normalization, util

LayerNorm = nn.LayerNorm
RMSNorm = normalization.RMSNorm
DecodeCache = util.DecodeCache


def make_attn_mask(
  query_length: int,
  num_all_masked_kv: torch.Tensor,
  query_index_offset: torch.Tensor | None = None,
  kv_length: int = 0,
) -> torch.Tensor:
  """Makes attention mask."""
  if kv_length == 0:
    kv_length = query_length

  q_index = torch.arange(query_length, device=num_all_masked_kv.device)[
    None, None, :, None
  ]
  if query_index_offset is not None:
    q_index = q_index + query_index_offset[:, None, None, None]
  kv_index = torch.arange(kv_length, device=num_all_masked_kv.device)[
    None, None, None, :
  ]
  return torch.logical_and(
    q_index >= kv_index,
    kv_index >= num_all_masked_kv[:, None, None, None],
  )


class RotaryPositionalEmbedding(nn.Module):
  """Rotary positional embedding."""

  def __init__(
    self,
    embedding_dims: int,
    min_timescale: float = 1.0,
    max_timescale: float = 10000.0,
  ):
    super().__init__()
    self.embedding_dims = embedding_dims
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale

  def forward(
    self,
    inputs: torch.Tensor,
    position: torch.Tensor | None = None,
  ):
    """Generates a JTensor of sinusoids with different frequencies."""
    if self.embedding_dims != inputs.shape[-1]:
      raise ValueError(
        "The embedding dims of the rotary position embedding"
        "must match the hidden dimension of the inputs."
      )
    half_embedding_dim = self.embedding_dims // 2
    fraction = (
      2
      * torch.arange(0, half_embedding_dim, device=inputs.device)
      / self.embedding_dims
    )
    timescale = (
      self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    ).to(inputs.device)
    if position is None:
      seq_length = inputs.shape[1]
      position = torch.arange(seq_length, dtype=torch.float32, device=inputs.device)[
        None, :
      ]

    if len(inputs.shape) == 4:
      position = position[..., None, None]
      timescale = timescale[None, None, None, :]
    elif len(inputs.shape) == 3:
      position = position[..., None]
      timescale = timescale[None, None, :]
    else:
      raise ValueError("Inputs must be of rank 3 or 4.")

    sinusoid_inp = position / timescale
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    first_half, second_half = torch.chunk(inputs, 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return torch.cat([first_part, second_part], dim=-1)


def _dot_product_attention(
  query,
  key,
  value,
  mask=None,
):
  """Computes dot-product attention given query, key, and value."""
  attn_weights = torch.einsum("...qhd,...khd->...hqk", query, key)
  if mask is not None:
    attn_weights = torch.where(
      mask, attn_weights, -torch.finfo(attn_weights.dtype).max / 2
    )

  attn_weights = F.softmax(attn_weights, dim=-1)

  return torch.einsum("...hqk,...khd->...qhd", attn_weights, value)


def _torch_dot_product_attention(query, key, value, mask=None):
  """
  Performs the exact same (unscaled) attention as the above function,
  but using the fast and fused F.scaled_dot_product_attention kernel.
  """

  # 1. Permute inputs from (B, L, H, D) to the expected (B, H, L, D)
  query = query.permute(0, 2, 1, 3)
  key = key.permute(0, 2, 1, 3)
  value = value.permute(0, 2, 1, 3)

  # 2. Call the fused attention kernel
  #    - Pass the mask to `attn_mask`.
  #    - Set `scale=1.0` to disable the default 1/sqrt(d_k) scaling.
  output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=1.0)

  # 3. Permute the output back to the original (B, L, H, D) layout
  output = output.permute(0, 2, 1, 3)

  return output


class PerDimScale(nn.Module):
  """Per-dimension scaling."""

  def __init__(self, num_dims: int):
    super().__init__()
    self.num_dims = num_dims
    self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    scale_factor = (
      1.442695041 / math.sqrt(self.num_dims) * F.softplus(self.per_dim_scale)
    )
    return x * scale_factor


class MultiHeadAttention(nn.Module):
  """Multi-head attention."""

  def __init__(
    self,
    num_heads: int,
    in_features: int,
    *,
    use_per_dim_scale: bool = True,
    use_rotary_position_embeddings: bool = True,
    use_bias: bool = False,
    attention_fn: Callable[..., torch.Tensor] = _torch_dot_product_attention,
    qk_norm: str = "rms",
    fuse_qkv: bool = False,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.in_features = in_features
    self.head_dim = in_features // num_heads
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.qk_norm = qk_norm
    self.fuse_qkv = fuse_qkv

    if self.in_features % self.num_heads != 0:
      raise ValueError(
        f"Memory dimension ({self.in_features}) must be divisible by "
        f"'num_heads' heads ({self.num_heads})."
      )

    if self.fuse_qkv:
      self.qkv_proj = nn.Linear(self.in_features, 3 * self.in_features, bias=use_bias)
    else:
      self.query = nn.Linear(self.in_features, self.in_features, bias=use_bias)
      self.key = nn.Linear(self.in_features, self.in_features, bias=use_bias)
      self.value = nn.Linear(self.in_features, self.in_features, bias=use_bias)
    self.out = nn.Linear(self.in_features, self.in_features, bias=use_bias)

    if self.qk_norm == "rms":
      self.query_ln = RMSNorm(self.head_dim)
      self.key_ln = RMSNorm(self.head_dim)
    else:
      self.query_ln = nn.Identity()
      self.key_ln = nn.Identity()

    self.use_rotary_position_embeddings = use_rotary_position_embeddings
    if self.use_rotary_position_embeddings:
      self.rotary_position_embedding = RotaryPositionalEmbedding(
        embedding_dims=self.head_dim,
      )

    self.use_per_dim_scale = use_per_dim_scale
    if use_per_dim_scale:
      self.per_dim_scale = PerDimScale(num_dims=self.head_dim)

  def forward(
    self,
    inputs_q: torch.Tensor,
    *,
    decode_cache: DecodeCache | None = None,
    patch_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, DecodeCache | None]:
    b, n_patches, _ = inputs_q.shape
    if patch_mask is None:
      patch_mask = torch.zeros(b, n_patches, dtype=torch.bool, device=inputs_q.device)

    if self.fuse_qkv:
      qkv = self.qkv_proj(inputs_q)
      query, key, value = torch.chunk(qkv, 3, dim=-1)
      query = query.view(b, n_patches, self.num_heads, self.head_dim)
      key = key.view(b, n_patches, self.num_heads, self.head_dim)
      value = value.view(b, n_patches, self.num_heads, self.head_dim)
    else:
      query = self.query(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
      key = self.key(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
      value = self.value(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)

    if decode_cache is None:
      num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)
      next_index = torch.zeros_like(num_masked, dtype=torch.int32)
    else:
      num_masked = (
        torch.sum(patch_mask.to(torch.int32), dim=-1) + decode_cache.num_masked
      )
      next_index = decode_cache.next_index.clone()

    if self.use_rotary_position_embeddings:
      position = (
        torch.arange(n_patches, device=inputs_q.device)[None, :]
        + next_index[:, None]
        - num_masked[:, None]
      )
      query = self.rotary_position_embedding(query, position)
      key = self.rotary_position_embedding(key, position)

    query = self.query_ln(query)
    key = self.key_ln(key)

    if self.use_per_dim_scale:
      query = self.per_dim_scale(query)

    if decode_cache is not None:
      _, decode_cache_size, _, _ = decode_cache.value.shape

      start = decode_cache.next_index[0]
      end = start + n_patches

      # Perform a single, vectorized slice assignment for the entire batch.
      # This is vastly more efficient than a Python for-loop.

      decode_cache.key[:, start:end] = key
      decode_cache.value[:, start:end] = value

      key = decode_cache.key
      value = decode_cache.value
      decode_cache.next_index += n_patches
      decode_cache.num_masked = num_masked
      attn_mask = make_attn_mask(
        query_length=n_patches,
        num_all_masked_kv=num_masked,
        query_index_offset=next_index,
        kv_length=decode_cache_size,
      )
    else:
      attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)

    x = self.attention_fn(
      query,
      key,
      value,
      mask=attn_mask,
    )

    x = x.reshape(b, n_patches, self.in_features)
    out = self.out(x)
    return out, decode_cache


class Transformer(nn.Module):
  """Classic Transformer used in TimesFM."""

  def __init__(self, config: configs.TransformerConfig):
    super().__init__()
    self.config = config

    if config.attention_norm == "rms":
      self.pre_attn_ln = RMSNorm(num_features=config.model_dims)
      self.post_attn_ln = RMSNorm(num_features=config.model_dims)
    else:
      raise ValueError(f"Layer norm: {config.attention_norm} not supported.")

    self.attn = MultiHeadAttention(
      num_heads=config.num_heads,
      in_features=config.model_dims,
      use_per_dim_scale=True,
      use_rotary_position_embeddings=config.use_rotary_position_embeddings,
      qk_norm=config.qk_norm,
      fuse_qkv=config.fuse_qkv,
    )

    if config.feedforward_norm == "rms":
      self.pre_ff_ln = RMSNorm(num_features=config.model_dims)
      self.post_ff_ln = RMSNorm(num_features=config.model_dims)
    else:
      raise ValueError(f"Layer norm: {config.feedforward_norm} not supported.")

    self.ff0 = nn.Linear(
      in_features=config.model_dims,
      out_features=config.hidden_dims,
      bias=config.use_bias,
    )
    self.ff1 = nn.Linear(
      in_features=config.hidden_dims,
      out_features=config.model_dims,
      bias=config.use_bias,
    )
    if config.ff_activation == "relu":
      self.activation = nn.ReLU()
    elif config.ff_activation == "swish":
      self.activation = nn.SiLU()
    elif config.ff_activation == "none":
      self.activation = nn.Identity()
    else:
      raise ValueError(f"Activation: {config.ff_activation} not supported.")

  def forward(
    self,
    input_embeddings: torch.Tensor,
    patch_mask: torch.Tensor,
    decode_cache: DecodeCache | None = None,
  ) -> tuple[torch.Tensor, DecodeCache | None]:
    attn_output, decode_cache = self.attn(
      inputs_q=self.pre_attn_ln(input_embeddings),
      decode_cache=decode_cache,
      patch_mask=patch_mask,
    )
    attn_output = self.post_attn_ln(attn_output) + input_embeddings
    output_embeddings = (
      self.post_ff_ln(self.ff1(self.activation(self.ff0(self.pre_ff_ln(attn_output)))))
      + attn_output
    )
    return output_embeddings, decode_cache
