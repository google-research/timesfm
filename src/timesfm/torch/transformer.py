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
from typing import Callable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .. import abstract
from . import normalization
from . import util

LayerNorm = nn.LayerNorm
RMSNorm = normalization.RMSNorm
TransformerConfig = abstract.TransformerConfig
DecodeCache = util.DecodeCache


def make_attn_mask(
    query_length: int,
    num_all_masked_kv: torch.Tensor,
    query_index_offset: Optional[torch.Tensor] = None,
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
      position: Optional[torch.Tensor] = None,
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
        self.min_timescale
        * (self.max_timescale / self.min_timescale) ** fraction
    ).to(inputs.device)
    if position is None:
      seq_length = inputs.shape[1]
      position = torch.arange(
          seq_length, dtype=torch.float32, device=inputs.device
      )[None, :]

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
      attention_fn: Callable[..., torch.Tensor] = _dot_product_attention,
      qk_norm: str = "rms",
  ):
    super().__init__()
    self.num_heads = num_heads
    self.in_features = in_features
    self.head_dim = in_features // num_heads
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.qk_norm = qk_norm

    if self.in_features % self.num_heads != 0:
      raise ValueError(
          f"Memory dimension ({self.in_features}) must be divisible by "
          f"'num_heads' heads ({self.num_heads})."
      )

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
      decode_cache: Optional[DecodeCache] = None,
      patch_mask: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, Optional[DecodeCache]]:
    b, n_patches, _ = inputs_q.shape
    if patch_mask is None:
      patch_mask = torch.zeros(
          b, n_patches, dtype=torch.bool, device=inputs_q.device
      )

    query = self.query(inputs_q).view(
        b, n_patches, self.num_heads, self.head_dim
    )
    key = self.key(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
    value = self.value(inputs_q).view(
        b, n_patches, self.num_heads, self.head_dim
    )

    if decode_cache is None:
      num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)
      next_index = torch.zeros_like(num_masked, dtype=torch.int32)
    else:
      num_masked = (
          torch.sum(patch_mask.to(torch.int32), dim=-1)
          + decode_cache.num_masked
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
      for i in range(b):
        start = decode_cache.next_index[i]
        end = start + n_patches
        decode_cache.key[i, start:end] = key[i].clone()
        decode_cache.value[i, start:end] = value[i].clone()
      key = decode_cache.key.clone()
      value = decode_cache.value.clone()
      decode_cache.next_index += n_patches
      decode_cache.num_masked += num_masked
      attn_mask = make_attn_mask(
          query_length=n_patches,
          num_all_masked_kv=decode_cache.num_masked,
          query_index_offset=next_index,
          kv_length=decode_cache_size,
      )
    else:
      attn_mask = make_attn_mask(
          query_length=n_patches, num_all_masked_kv=num_masked
      )

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

  def __init__(self, config: TransformerConfig):
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
      decode_cache: Optional[DecodeCache] = None,
  ) -> Tuple[torch.Tensor, Optional[DecodeCache]]:
    attn_output, decode_cache = self.attn(
        inputs_q=self.pre_attn_ln(input_embeddings),
        decode_cache=decode_cache,
        patch_mask=patch_mask,
    )
    attn_output = self.post_attn_ln(attn_output) + input_embeddings
    output_embeddings = (
        self.post_ff_ln(
            self.ff1(self.activation(self.ff0(self.pre_ff_ln(attn_output))))
        )
        + attn_output
    )
    return output_embeddings, decode_cache
