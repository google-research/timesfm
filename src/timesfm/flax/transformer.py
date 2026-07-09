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

import functools
from typing import Callable

from flax import nnx
from flax.nnx.nn import linear
import jax
from jax import lax
import jax.numpy as jnp
import jaxtyping

from .. import configs
from . import normalization, util

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
Num = jaxtyping.Num
LayerNorm = normalization.LayerNorm
RMSNorm = normalization.RMSNorm
LinearGeneral = linear.LinearGeneral
TransformerConfig = configs.TransformerConfig
DecodeCache = util.DecodeCache


@functools.partial(
  jax.jit,
  static_argnames=("query_length", "kv_length"),
)
def make_attn_mask(
  query_length: int,
  num_all_masked_kv: Integer[Array, "b"],
  query_index_offset: Integer[Array, "b"] | None = None,
  kv_length: int = 0,
) -> Bool[Array, "b 1 q n"]:
  """Makes attention mask."""

  if kv_length == 0:
    kv_length = query_length

  q_index = jnp.arange(query_length)[None, None, :, None]
  if query_index_offset is not None:
    q_index += query_index_offset[:, None, None, None]
  kv_index = jnp.arange(kv_length)[None, None, None, :]
  return jnp.logical_and(
    q_index >= kv_index,
    kv_index >= num_all_masked_kv[:, None, None, None],
  )


class RotaryPositionalEmbedding(nnx.Module):
  """Rotary positional embedding."""

  def __init__(
    self,
    embedding_dims: int,
    min_timescale: int = 1,
    max_timescale: int = 10000,
  ):
    self.embedding_dims = embedding_dims
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale

  def __call__(
    self,
    inputs: Float[Array, "b ... d"],
    position: Array | None = None,
  ):
    """Generates a JTensor of sinusoids with different frequencies."""
    if self.embedding_dims != inputs.shape[-1]:
      raise ValueError(
        "The embedding dims of the rotary position embedding"
        "must match the hidden dimension of the inputs."
      )
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    timescale = (
      self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    )
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[None, :]
    if len(inputs.shape) == 4:
      position = position[..., None, None]
      timescale = timescale[None, None, None, :]
    elif len(inputs.shape) == 3:
      position = position[..., None]
      timescale = timescale[None, None, :]
    else:
      raise ValueError("Inputs must be of rank 3 or 4.")
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    first_part = first_part.astype(None)
    second_part = second_part.astype(None)
    return jnp.concatenate([first_part, second_part], axis=-1)


class PerDimScale(nnx.Module):
  """Per-dimension scaling."""

  __data__ = ("per_dim_scale",)

  def __init__(self, num_dims: int, *, rngs=nnx.Rngs(42)):
    del rngs
    self.num_dims = num_dims
    self.per_dim_scale = nnx.Param(jnp.zeros(shape=(num_dims,)))

  def __call__(self, x: Float[Array, "b ... d"]) -> Float[Array, "b ... d"]:
    return x * (
      1.442695041 / jnp.sqrt(self.num_dims) * jax.nn.softplus(self.per_dim_scale)
    )


class MultiHeadAttention(nnx.Module):
  """Multi-head attention."""

  def __init__(
    self,
    num_heads: int,
    in_features: int,
    *,
    use_per_dim_scale: bool = True,
    use_rotary_position_embeddings: bool = True,
    use_bias: bool = False,
    deterministic: bool | None = None,
    attention_fn: Callable[..., Array] = nnx.dot_product_attention,
    qk_norm: str = "rms",
    rngs=nnx.Rngs(42),
  ):
    self.num_heads = num_heads
    self.in_features = in_features
    self.qkv_features = in_features
    self.out_features = in_features
    self.in_kv_features = in_features
    self.deterministic = deterministic
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.qk_norm = qk_norm

    if self.qkv_features % self.num_heads != 0:
      raise ValueError(
        f"Memory dimension ({self.qkv_features}) must be divisible by "
        f"'num_heads' heads ({self.num_heads})."
      )
    self.head_dim = self.qkv_features // self.num_heads

    linear_general = functools.partial(
      LinearGeneral,
      out_features=(self.num_heads, self.head_dim),
      use_bias=self.use_bias,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    self.query = linear_general(self.in_features, rngs=rngs)
    self.key = linear_general(self.in_kv_features, rngs=rngs)
    self.value = linear_general(self.in_kv_features, rngs=rngs)

    if self.qk_norm == "rms":
      self.query_ln = RMSNorm(self.head_dim)
      self.key_ln = RMSNorm(self.head_dim)
    else:
      self.query_ln = None
      self.key_ln = None

    self.out = LinearGeneral(
      in_features=(self.num_heads, self.head_dim),
      out_features=self.out_features,
      axis=(-2, -1),
      use_bias=self.use_bias,
      rngs=rngs,
    )

    self.use_per_dim_scale = use_per_dim_scale
    self.use_rotary_position_embeddings = use_rotary_position_embeddings
    if self.use_rotary_position_embeddings:
      self.rotary_position_embedding = RotaryPositionalEmbedding(
        embedding_dims=self.head_dim,
      )
    else:
      self.rotary_position_embedding = None

    if use_per_dim_scale:
      self.per_dim_scale = PerDimScale(num_dims=self.head_dim, rngs=rngs)
    else:
      self.per_dim_scale = None

  def __call__(
    self,
    inputs_q: Array,
    *,
    decode_cache: DecodeCache | None = None,
    patch_mask: Array | None = None,
    deterministic: bool | None = None,
    sow_weights: bool = False,
  ) -> tuple[Float[Array, "b ... o"], DecodeCache | None]:
    """Applies multi-head dot product attention on the input data."""
    _, n_patches, input_in_features = inputs_q.shape
    if input_in_features != self.in_features:
      raise ValueError(
        f"Incompatible input dimension, got {input_in_features} "
        f"but module expects {self.in_features}."
      )
    if patch_mask is None:
      patch_mask = jnp.zeros_like(inputs_q.shape[:-1], dtype=jnp.bool)

    # For query: rope -> ln -> per_dim_scale
    query = self.query(inputs_q)
    key = self.key(inputs_q)
    value = self.value(inputs_q)

    if decode_cache is None:
      num_masked = jnp.sum(patch_mask.astype(jnp.int32), axis=-1, keepdims=False)
      next_index = jnp.zeros_like(num_masked, dtype=jnp.int32)
    else:
      num_masked = (
        jnp.sum(patch_mask.astype(jnp.int32), axis=-1, keepdims=False)
        + decode_cache.num_masked
      )
      next_index = decode_cache.next_index

    if self.use_rotary_position_embeddings:
      position = (
        jnp.arange(n_patches, dtype=jnp.int32)[None, :]
        + next_index[:, None]
        - num_masked[:, None]
      )
      query = self.rotary_position_embedding(query, position)
      key = self.rotary_position_embedding(key, position)
    if self.query_ln is not None:
      query = self.query_ln(query)
    if self.key_ln is not None:
      key = self.key_ln(key)
    if self.use_per_dim_scale:
      query = self.per_dim_scale(query)

    if decode_cache is not None:
      # Cached decoding.
      _, decode_cache_size, _, _ = decode_cache.value.shape
      zero = jnp.array(0, dtype=lax.dtype(next_index.dtype))
      start_indices = (zero, next_index[0], zero, zero)
      key = lax.dynamic_update_slice(decode_cache.key, key, start_indices)
      value = lax.dynamic_update_slice(decode_cache.value, value, start_indices)
      decode_cache.key = key
      decode_cache.value = value
      decode_cache.next_index = next_index + n_patches
      decode_cache.num_masked = num_masked
      attn_mask = make_attn_mask(
        query_length=n_patches,
        num_all_masked_kv=num_masked,
        query_index_offset=next_index,
        kv_length=decode_cache_size,
      )
    else:
      # Training
      attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)

    # apply attention
    x = self.attention_fn(
      query * jnp.sqrt(self.head_dim),
      key,
      value,
      mask=attn_mask,
      deterministic=deterministic,
      module=self if sow_weights else None,
    )
    # back to the original inputs dimensions
    out = self.out(x)
    return out, decode_cache


class Transformer(nnx.Module):
  """Classic Transformer used in TimesFM."""

  def __init__(self, config: TransformerConfig, *, rngs=nnx.Rngs(42)):
    self.config = config

    if config.attention_norm == "rms":
      self.pre_attn_ln = RMSNorm(num_features=config.model_dims, rngs=rngs)
      self.post_attn_ln = RMSNorm(num_features=config.model_dims, rngs=rngs)
    else:
      raise ValueError(f"Layer norm: {config.attention_norm} not supported.")

    self.attn = MultiHeadAttention(
      num_heads=config.num_heads,
      in_features=config.model_dims,
      use_per_dim_scale=True,
      use_rotary_position_embeddings=config.use_rotary_position_embeddings,
      qk_norm=config.qk_norm,
      rngs=rngs,
    )

    if config.feedforward_norm == "rms":
      self.pre_ff_ln = RMSNorm(num_features=config.model_dims, rngs=rngs)
      self.post_ff_ln = RMSNorm(num_features=config.model_dims, rngs=rngs)
    else:
      raise ValueError(f"Layer norm: {config.feedforward_norm} not supported.")
    self.ff0 = nnx.Linear(
      in_features=config.model_dims,
      out_features=config.hidden_dims,
      use_bias=config.use_bias,
      rngs=rngs,
    )
    self.ff1 = nnx.Linear(
      in_features=config.hidden_dims,
      out_features=config.model_dims,
      use_bias=config.use_bias,
      rngs=rngs,
    )
    if config.ff_activation == "relu":
      self.activation = jax.nn.relu
    elif config.ff_activation == "swish":
      self.activation = jax.nn.swish
    elif config.ff_activation == "none":
      self.activation = lambda x: x
    else:
      raise ValueError(f"Activation: {config.ff_activation} not supported.")

  def __call__(
    self,
    input_embeddings: Float[Array, "b n d"],
    patch_mask: Bool[Array, "b n"],
    decode_cache: DecodeCache | None = None,
  ) -> tuple[Float[Array, "b n d"], DecodeCache | None]:
    attn_output, decode_cache = self.attn(
      inputs_q=self.pre_attn_ln(input_embeddings),
      decode_cache=decode_cache,
      patch_mask=patch_mask,
      sow_weights=False,
      deterministic=True,
    )
    attn_output = self.post_attn_ln(attn_output) + input_embeddings
    output_embeddings = (
      self.post_ff_ln(self.ff1(self.activation(self.ff0(self.pre_ff_ln(attn_output)))))
      + attn_output
    )
    return output_embeddings, decode_cache
