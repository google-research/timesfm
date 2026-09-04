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

"""Transformer layers for TimesFM3 PyTorch (inference only).

Port of the Flax MixingTransformer architecture:
  - RotaryPositionalEmbedding
  - MultiHeadAttention (with KV-cache support)
  - MixingTransformer (sequential seq + variate attention + FFN)
  - StackedMixingTransformer (nn.ModuleList of MixingTransformer)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import configs, normalization, util


def make_attn_mask(
  query_length: int,
  num_all_masked_kv: torch.Tensor,
  query_index_offset: torch.Tensor | None = None,
  kv_length: int = 0,
  causal: bool = True,
) -> torch.Tensor:
  """Makes attention mask. True = attend, False = mask.

  Args:
    query_length: Number of query positions.
    num_all_masked_kv: Shape (b,). Number of leading masked KV positions.
    query_index_offset: Shape (b,). Offset for query indices (decode mode).
    kv_length: Length of KV sequence. Defaults to query_length.
    causal: Whether to apply causal masking.

  Returns:
    Boolean mask of shape (b, 1, query_length, kv_length). True = attend.
  """
  if kv_length == 0:
    kv_length = query_length

  device = num_all_masked_kv.device
  q_index = torch.arange(query_length, device=device).view(1, 1, -1, 1)
  if query_index_offset is not None:
    q_index = q_index + query_index_offset.view(-1, 1, 1, 1)
  kv_index = torch.arange(kv_length, device=device).view(1, 1, 1, -1)
  mask = kv_index >= num_all_masked_kv.view(-1, 1, 1, 1)
  if causal:
    return (q_index >= kv_index) & mask
  return mask


def make_segment_mask(
  segment_ids: torch.Tensor,
) -> torch.Tensor:
  """Makes a segment mask from segment ids.

  Args:
    segment_ids: Shape (b, seq_length).

  Returns:
    Boolean mask of shape (b, 1, seq_length, seq_length).
  """
  return (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)


class RotaryPositionalEmbedding(nn.Module):
  """Rotary positional embedding (RoPE).

  Stateless module — no learnable parameters.
  Supports 3D (b, n, d) and 4D (b, n, h, hd) inputs.
  """

  def __init__(
    self,
    embedding_dims: int,
    min_timescale: int = 1,
    max_timescale: int = 10000,
  ):
    super().__init__()
    self.embedding_dims = embedding_dims
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale

    half_dim = embedding_dims // 2
    fraction = 2.0 * torch.arange(half_dim, dtype=torch.float32) / embedding_dims
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    self.register_buffer("timescale", timescale, persistent=False)

  def forward(
    self,
    inputs: torch.Tensor,
    position: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Applies rotary positional embeddings.

    Args:
      inputs: Shape (b, n, d) or (b, n, h, hd).
      position: Shape (b, n). If None, uses arange(n).

    Returns:
      Tensor with same shape as inputs, with RoPE applied.
    """
    if self.embedding_dims != inputs.shape[-1]:
      raise ValueError(
        "The embedding dims of the rotary position embedding "
        "must match the hidden dimension of the inputs."
      )
    timescale = self.timescale.to(inputs.device)

    if position is None:
      seq_length = inputs.shape[1]
      position = torch.arange(
        seq_length, device=inputs.device, dtype=torch.float32
      ).unsqueeze(0)

    if inputs.dim() == 4:
      # (b, n) -> (b, n, 1, 1) for broadcasting with (b, n, h, hd)
      pos = position.unsqueeze(-1).unsqueeze(-1)
      ts = timescale.view(1, 1, 1, -1)
    elif inputs.dim() == 3:
      pos = position.unsqueeze(-1)
      ts = timescale.view(1, 1, -1)
    else:
      raise ValueError("Inputs must be of rank 3 or 4.")

    sinusoid_inp = pos.float() / ts
    sin_val = torch.sin(sinusoid_inp)
    cos_val = torch.cos(sinusoid_inp)
    first_half, second_half = inputs.chunk(2, dim=-1)
    first_part = first_half * cos_val - second_half * sin_val
    second_part = second_half * cos_val + first_half * sin_val
    return torch.cat([first_part, second_part], dim=-1)


class MultiHeadAttention(nn.Module):
  """Multi-head attention with RoPE, QK-norm, PerDimScale, and KV-cache.

  This matches the Flax MultiHeadAttention exactly, including the
  pre-multiplication of query by sqrt(head_dim) which cancels with the
  standard 1/sqrt(d) scaling in dot-product attention.
  """

  def __init__(
    self,
    num_heads: int,
    in_features: int,
    use_per_dim_scale: bool = True,
    use_rotary_position_embeddings: bool = True,
    causal_attention: bool = True,
    use_bias: bool = False,
    qk_norm: str = "rms",
    v_norm: str = "none",
    use_sdpa: bool = False,
    rescale_logits: bool = False,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.in_features = in_features
    self.causal_attention = causal_attention
    self.head_dim = in_features // num_heads
    self.use_sdpa = use_sdpa
    # rescale_logits=False → MEA=True behaviour: Q is pre-multiplied by √d,
    #   no internal division. Matches Flax
    #   memory_efficient_attention(rescale_logits=False).
    # rescale_logits=True  → MEA=False behaviour: Q*√d is passed but divided
    #   by √d internally, so they cancel (net scale = 1.0). Matches Flax
    #   nn.dot_product_attention.
    self.rescale_logits = rescale_logits

    # Q, K, V projections: Linear(in, heads*hd)
    # We'll reshape the output to (b, n, heads, hd)
    self.query_proj = nn.Linear(in_features, in_features, bias=use_bias)
    self.key_proj = nn.Linear(in_features, in_features, bias=use_bias)
    self.value_proj = nn.Linear(in_features, in_features, bias=use_bias)

    # Output projection
    self.out_proj = nn.Linear(in_features, in_features, bias=use_bias)

    # QK normalization
    if qk_norm == "rms":
      self.query_ln = nn.RMSNorm(self.head_dim)
      self.key_ln = nn.RMSNorm(self.head_dim)
    else:
      self.query_ln = None
      self.key_ln = None

    # V normalization
    if v_norm == "rms":
      self.value_ln = nn.RMSNorm(self.head_dim, elementwise_affine=False)
    else:
      self.value_ln = None

    # RoPE
    if use_rotary_position_embeddings:
      self.rotary_position_embedding = RotaryPositionalEmbedding(
        embedding_dims=self.head_dim
      )
    else:
      self.rotary_position_embedding = None

    # PerDimScale
    if use_per_dim_scale:
      self.per_dim_scale = normalization.PerDimScale(num_dims=self.head_dim)
    else:
      self.per_dim_scale = None

  def forward(
    self,
    inputs_q: torch.Tensor,
    *,
    segment_ids: torch.Tensor | None = None,
    segment_pos: torch.Tensor | None = None,
    decode_cache: util.DecodeCache | None = None,
    patch_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, util.DecodeCache | None, torch.Tensor]:
    """Applies multi-head attention.

    Args:
      inputs_q: Shape (b, n, d).
      segment_ids: Shape (b, n). Optional segment IDs for masking.
      segment_pos: Shape (b, n). Optional positions for RoPE.
      decode_cache: Optional KV cache for autoregressive decoding.
      patch_mask: Shape (b, n). True = masked patch.

    Returns:
      Tuple of (output, updated_cache, attn_mask).
      - output: Shape (b, n, d).
      - updated_cache: Updated DecodeCache or None.
      - attn_mask: The attention mask used.
    """
    batch_size, n_patches, _ = inputs_q.shape
    device = inputs_q.device

    if patch_mask is None:
      patch_mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)

    # Project Q, K, V and reshape to (b, n, h, hd)
    query = self.query_proj(inputs_q).view(
      batch_size, n_patches, self.num_heads, self.head_dim
    )
    key = self.key_proj(inputs_q).view(
      batch_size, n_patches, self.num_heads, self.head_dim
    )
    value = self.value_proj(inputs_q).view(
      batch_size, n_patches, self.num_heads, self.head_dim
    )

    if decode_cache is None:
      num_front_masked = torch.sum(torch.cumprod(patch_mask.int(), dim=-1), dim=-1)
      next_index = torch.zeros_like(num_front_masked, dtype=torch.int32)
    else:
      num_front_masked = decode_cache.num_front_masked
      next_index = decode_cache.next_index

    # Apply RoPE
    if self.rotary_position_embedding is not None:
      if segment_pos is None:
        position = torch.arange(n_patches, device=device, dtype=torch.int32).unsqueeze(
          0
        ) + next_index.unsqueeze(-1)
      else:
        position = segment_pos
      query = self.rotary_position_embedding(query, position)
      key = self.rotary_position_embedding(key, position)

    # QK normalization
    if self.query_ln is not None:
      query = self.query_ln(query)
    if self.key_ln is not None:
      key = self.key_ln(key)

    # PerDimScale
    if self.per_dim_scale is not None:
      query = self.per_dim_scale(query)

    # V normalization
    if self.value_ln is not None:
      value = self.value_ln(value)

    if decode_cache is not None:
      # Cached decoding: update cache with new K, V
      cache_size = decode_cache.key.shape[1]
      if torch.all(next_index == next_index[0]):
        idx = next_index[0].item()
        decode_cache.key[:, idx : idx + n_patches, :, :] = key
        decode_cache.value[:, idx : idx + n_patches, :, :] = value
      else:
        for b_idx in range(batch_size):
          idx_b = next_index[b_idx].item()
          decode_cache.key[b_idx, idx_b : idx_b + n_patches, :, :] = key[b_idx]
          decode_cache.value[b_idx, idx_b : idx_b + n_patches, :, :] = value[b_idx]
      key = decode_cache.key
      value = decode_cache.value

      decode_cache = util.DecodeCache(
        next_index=next_index + n_patches,
        num_front_masked=num_front_masked,
        key=key,
        value=value,
      )

      attn_mask = make_attn_mask(
        query_length=n_patches,
        num_all_masked_kv=num_front_masked,
        query_index_offset=next_index,
        kv_length=cache_size,
        causal=self.causal_attention,
      )
    else:
      # Training / full-sequence mode
      attn_mask = make_attn_mask(
        query_length=n_patches,
        num_all_masked_kv=torch.zeros_like(num_front_masked),
        causal=self.causal_attention,
      )
      # Apply patch_mask to K/V positions
      attn_mask = attn_mask & (~patch_mask[:, None, None, :])
      if segment_ids is not None:
        segment_mask = make_segment_mask(segment_ids)
        attn_mask = attn_mask & segment_mask

    # Transpose for attention: (b, n, h, d) -> (b, h, n, d)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if self.use_sdpa:
      # --- F.scaled_dot_product_attention path (PyTorch >= 2.1) ---
      # SDPA computes: softmax(Q @ K^T * scale) @ V.
      if self.rescale_logits:
        # MEA=False equivalent: Flax passes Q*√d to nn.dot_product_attention
        # which divides by √d internally → net scale = 1.0.
        attn_scale = 1.0
      else:
        # MEA=True equivalent: Flax passes Q*√d with rescale_logits=False
        # → no internal division → net scale = √d.
        attn_scale = math.sqrt(self.head_dim)
      x = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.expand(-1, self.num_heads, -1, -1),
        scale=attn_scale,
      )
    else:
      # --- Manual attention path ---
      # Convert mask: True=attend -> 0.0, False=mask -> -1e9 (matching Flax
      # MEA bias).
      float_mask = torch.where(
        attn_mask.expand(-1, self.num_heads, -1, -1),
        torch.tensor(0.0, device=device),
        torch.tensor(-1e9, device=device),
      )
      if self.rescale_logits:
        # MEA=False equivalent: Q*√d then divide by √d → net scale = 1.0.
        # Flax nn.dot_product_attention receives Q*√d and divides by √d
        # internally, so the pre-multiplication and rescaling cancel out.
        query = query * math.sqrt(self.head_dim)
        attn_logits = (
          torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
          + float_mask
        )
      else:
        # MEA=True equivalent: Q*√d, no internal division → net scale = √d.
        # Matches Flax memory_efficient_attention(rescale_logits=False).
        query = query * math.sqrt(self.head_dim)
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) + float_mask
      attn_weights = F.softmax(attn_logits, dim=-1)
      x = torch.matmul(attn_weights, value)

    # Transpose back: (b, h, n, d) -> (b, n, h, d)
    x = x.transpose(1, 2).contiguous()

    # Reshape and project: (b, n, h, d) -> (b, n, h*d)
    x = x.view(batch_size, n_patches, self.in_features)

    out = self.out_proj(x)
    return out, decode_cache, attn_mask


class MixingTransformer(nn.Module):
  """Transformer with sequential sequence and variate attention.

  Attention is applied first across the sequence dimension 'n', then
  across the variate dimension 'v' for inputs of shape 'b v n d'.

  Architecture per layer:
    1. Sequence attention: pre_ln -> reshape(bv,n,d) -> MHA -> post_ln +
    residual
    2. Variate attention: pre_ln -> reshape(bn,v,d) -> MHA -> post_ln + residual
    3. FFN: pre_ln -> ff0 -> activation -> ff1 -> post_ln + residual
  """

  def __init__(
    self,
    config: configs.TransformerConfig,
    use_variate_attention: bool = True,
  ):
    super().__init__()
    self.config = config
    self.use_variate_attention = use_variate_attention

    # Sequence attention norms + module
    self.pre_seq_attn_ln = nn.RMSNorm(config.model_dims)
    self.post_seq_attn_ln = nn.RMSNorm(config.model_dims)
    # rescale_logits mirrors Flax: use_memory_efficient_attention=True →
    #   MEA=True (rescale_logits=False, scale=√d);
    #   use_memory_efficient_attention=False → MEA=False (rescale_logits=True,
    #   net scale=1.0).
    rescale_logits = not getattr(config, "use_memory_efficient_attention", False)
    self.seq_attn = MultiHeadAttention(
      num_heads=config.num_heads,
      in_features=config.model_dims,
      use_per_dim_scale=True,
      use_rotary_position_embeddings=config.use_rope_seq,
      qk_norm=config.qk_norm,
      v_norm=getattr(config, "v_norm", "none"),
      causal_attention=getattr(config, "causal_attention", True),
      use_bias=config.use_bias,
      use_sdpa=config.use_sdpa,
      rescale_logits=rescale_logits,
    )

    # Variate attention norms + module
    if use_variate_attention:
      self.pre_var_attn_ln = nn.RMSNorm(config.model_dims)
      self.post_var_attn_ln = nn.RMSNorm(config.model_dims)
      self.var_attn = MultiHeadAttention(
        num_heads=config.num_heads,
        in_features=config.model_dims,
        use_per_dim_scale=True,
        use_rotary_position_embeddings=config.use_rope_var,
        qk_norm=config.qk_norm,
        v_norm=getattr(config, "v_norm", "none"),
        causal_attention=False,
        use_bias=config.use_bias,
        use_sdpa=config.use_sdpa,
        rescale_logits=rescale_logits,
      )

    # FFN norms + layers
    self.pre_ff_ln = nn.RMSNorm(config.model_dims)
    self.post_ff_ln = nn.RMSNorm(config.model_dims)
    self.ff0 = nn.Linear(config.model_dims, config.hidden_dims, bias=config.use_bias)
    self.ff1 = nn.Linear(config.hidden_dims, config.model_dims, bias=config.use_bias)
    self.activation = util.get_activation_fn(config.ff_activation)

  def forward(
    self,
    input_embeddings: torch.Tensor,
    patch_mask: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    segment_pos: torch.Tensor | None = None,
    decode_cache: util.DecodeCache | None = None,
    var_segment_pos: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, util.DecodeCache | None, torch.Tensor]:
    """Forward pass.

    Args:
      input_embeddings: Shape (b, v, n, d).
      patch_mask: Shape (b, v, n). True = masked.
      segment_ids: Shape (b, n). Optional.
      segment_pos: Shape (b, n). Optional.
      decode_cache: Optional KV cache.
      var_segment_pos: Shape (b*n, v). Optional variate positions.

    Returns:
      (output_embeddings, updated_cache, seq_attn_mask).
    """
    b, v, n, d = input_embeddings.shape

    # --- Sequence Attention ---
    seq_attn_in = self.pre_seq_attn_ln(input_embeddings)
    # (b, v, n, d) -> (b*v, n, d)
    seq_attn_in_flat = seq_attn_in.reshape(b * v, n, d)
    patch_mask_flat = patch_mask.reshape(b * v, n)

    # Broadcast segment_ids/pos across variates
    seq_seg_ids_flat = None
    if segment_ids is not None:
      seg_ids_bvn = segment_ids.unsqueeze(1).expand(b, v, n)
      seq_seg_ids_flat = seg_ids_bvn.reshape(b * v, n)

    seq_seg_pos_flat = None
    if segment_pos is not None:
      seg_pos_bvn = segment_pos.unsqueeze(1).expand(b, v, n)
      seq_seg_pos_flat = seg_pos_bvn.reshape(b * v, n)

    seq_attn_out_flat, decode_cache, seq_attn_mask = self.seq_attn(
      seq_attn_in_flat,
      segment_ids=seq_seg_ids_flat,
      segment_pos=seq_seg_pos_flat,
      decode_cache=decode_cache,
      patch_mask=patch_mask_flat,
    )
    seq_attn_out = seq_attn_out_flat.view(b, v, n, d)
    h1 = self.post_seq_attn_ln(seq_attn_out) + input_embeddings

    # --- Variate Attention ---
    if self.use_variate_attention:
      var_attn_in = self.pre_var_attn_ln(h1)
      # (b, v, n, d) -> (b*n, v, d)
      var_attn_in_flat = var_attn_in.permute(0, 2, 1, 3).reshape(b * n, v, d)
      # Mask: (b, v, n) -> (b, n, v) -> (b*n, v)
      var_patch_mask = patch_mask.permute(0, 2, 1).reshape(b * n, v)

      var_attn_out_flat, _, _ = self.var_attn(
        var_attn_in_flat,
        segment_pos=var_segment_pos,
        decode_cache=None,
        patch_mask=var_patch_mask,
      )
      # (b*n, v, d) -> (b, n, v, d) -> (b, v, n, d)
      var_attn_out = var_attn_out_flat.view(b, n, v, d).permute(0, 2, 1, 3)
      h2 = self.post_var_attn_ln(var_attn_out) + h1
    else:
      h2 = h1

    # --- FeedForward ---
    ff_out = self.ff1(self.activation(self.ff0(self.pre_ff_ln(h2))))
    output_embeddings = self.post_ff_ln(ff_out) + h2

    return output_embeddings, decode_cache, seq_attn_mask


class StackedMixingTransformer(nn.Module):
  """Stacked MixingTransformer layers."""

  def __init__(
    self,
    config: configs.StackedTransformersConfig,
    use_variate_attention: bool = True,
  ):
    super().__init__()
    self.config = config
    self.layers = nn.ModuleList(
      [
        MixingTransformer(
          config=config.transformer,
          use_variate_attention=use_variate_attention,
        )
        for _ in range(config.num_layers)
      ]
    )

  def forward(
    self,
    input_embeddings: torch.Tensor,
    patch_mask: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    segment_pos: torch.Tensor | None = None,
    decode_cache: list[util.DecodeCache] | None = None,
    var_segment_pos: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, list[util.DecodeCache] | None, list[torch.Tensor]]:
    """Forward pass through all layers.

    Args:
      input_embeddings: Shape (b, v, n, d).
      patch_mask: Shape (b, v, n).
      segment_ids: Shape (b, n). Optional.
      segment_pos: Shape (b, n). Optional.
      decode_cache: List of DecodeCache (one per layer), or None.
      var_segment_pos: Shape (b*n, v). Optional.

    Returns:
      (output_embeddings, updated_caches, attn_masks).
    """
    if decode_cache is None:
      decode_cache = [None] * len(self.layers)  # pyrefly: ignore[bad-assignment]

    output = input_embeddings
    new_caches = []
    attn_masks = []

    for i, layer in enumerate(self.layers):
      output, layer_cache, layer_mask = layer(
        output,
        patch_mask,
        segment_ids,
        segment_pos,
        decode_cache[i],  # pyrefly: ignore[unsupported-operation]
        var_segment_pos,
      )
      new_caches.append(layer_cache)
      attn_masks.append(layer_mask)

    return output, new_caches, attn_masks
