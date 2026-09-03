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

"""Tests for TimesFM3 PyTorch transformer layers."""

import unittest

import torch

from . import configs
from . import transformer as torch_trans
from . import util as torch_util


class MaskTest(unittest.TestCase):
  def test_make_attn_mask_causal(self):
    query_len = 4
    num_masked = torch.tensor([1])
    mask = torch_trans.make_attn_mask(
      query_length=query_len, num_all_masked_kv=num_masked, causal=True
    )
    # Shape: (1, 1, 4, 4)
    self.assertEqual(mask.shape, (1, 1, 4, 4))
    # Position 0 KV is masked (index 0 < 1)
    self.assertFalse(mask[0, 0, 0, 0].item())
    # Position (1, 1): query=1 >= kv=1 and kv=1 >= 1 => True
    self.assertTrue(mask[0, 0, 1, 1].item())
    # Position (1, 2): causal mask (1 < 2) => False
    self.assertFalse(mask[0, 0, 1, 2].item())

  def test_make_segment_mask(self):
    segment_ids = torch.tensor([[0, 0, 1, 1]])
    mask = torch_trans.make_segment_mask(segment_ids)
    self.assertEqual(mask.shape, (1, 1, 4, 4))
    self.assertTrue(mask[0, 0, 0, 1].item())
    self.assertFalse(mask[0, 0, 0, 2].item())


class RotaryEmbeddingTest(unittest.TestCase):
  def test_rope_forward(self):
    rope = torch_trans.RotaryPositionalEmbedding(embedding_dims=16)
    x = torch.randn(2, 8, 16)
    out = rope(x)
    self.assertEqual(out.shape, (2, 8, 16))

  def test_rope_with_position(self):
    rope = torch_trans.RotaryPositionalEmbedding(embedding_dims=16)
    x = torch.randn(2, 4, 16)
    positions = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]])
    out = rope(x, position=positions)
    self.assertEqual(out.shape, (2, 4, 16))


class MultiHeadAttentionTest(unittest.TestCase):
  def test_mha_forward_and_cache(self):
    mha = torch_trans.MultiHeadAttention(
      num_heads=4,
      in_features=64,
      qk_norm="rms",
      use_bias=False,
      use_sdpa=True,
    )
    b, n, d = 2, 8, 64
    x = torch.randn(b, n, d)
    patch_mask = torch.zeros(b, n, dtype=torch.bool)

    # Full forward pass
    out, cache, _ = mha(x, patch_mask=patch_mask)
    self.assertEqual(out.shape, (b, n, d))
    self.assertIsNone(cache)

    # Decode mode with cache
    decode_cache = torch_util.DecodeCache(
      next_index=torch.zeros(b, dtype=torch.int32),
      num_front_masked=torch.zeros(b, dtype=torch.int32),
      key=torch.zeros(b, 16, 4, 16),
      value=torch.zeros(b, 16, 4, 16),
    )
    q_x = torch.randn(b, 1, d)
    q_mask = torch.zeros(b, 1, dtype=torch.bool)
    out_step, updated_cache, _ = mha(q_x, patch_mask=q_mask, decode_cache=decode_cache)
    self.assertEqual(out_step.shape, (b, 1, d))
    self.assertIsNotNone(updated_cache)
    self.assertEqual(updated_cache.next_index[0].item(), 1)


class MixingTransformerTest(unittest.TestCase):
  def test_mixing_transformer_forward(self):
    cfg = configs.TransformerConfig(
      model_dims=64,
      hidden_dims=64,
      num_heads=4,
      attention_norm="rms",
      feedforward_norm="rms",
      qk_norm="rms",
      use_rope_seq=True,
      use_rope_var=True,
      use_bias=False,
      ff_activation="relu",
      deterministic=True,
      use_sdpa=True,
    )
    layer = torch_trans.MixingTransformer(config=cfg, use_variate_attention=True)
    b, v, n, d = 2, 3, 8, 64
    inputs = torch.randn(b, v, n, d)
    patch_mask = torch.zeros(b, v, n, dtype=torch.bool)

    out, cache, _ = layer(inputs, patch_mask=patch_mask)
    self.assertEqual(out.shape, (b, v, n, d))
    self.assertIsNone(cache)


class StackedMixingTransformerTest(unittest.TestCase):
  def test_stacked_transformer(self):
    sub_cfg = configs.TransformerConfig(
      model_dims=64,
      hidden_dims=64,
      num_heads=4,
      attention_norm="rms",
      feedforward_norm="rms",
      qk_norm="rms",
      use_rope_seq=True,
      use_rope_var=True,
      use_bias=False,
      ff_activation="relu",
      deterministic=True,
      use_sdpa=True,
    )
    stack_cfg = configs.StackedTransformersConfig(
      num_layers=2,
      use_remat=False,
      transformer=sub_cfg,
    )
    stack = torch_trans.StackedMixingTransformer(
      config=stack_cfg, use_variate_attention=True
    )
    b, v, n, d = 2, 2, 4, 64
    inputs = torch.randn(b, v, n, d)
    patch_mask = torch.zeros(b, v, n, dtype=torch.bool)

    out, _, _ = stack(inputs, patch_mask=patch_mask)
    self.assertEqual(out.shape, (b, v, n, d))


if __name__ == "__main__":
  unittest.main()
