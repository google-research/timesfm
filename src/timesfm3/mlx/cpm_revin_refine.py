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

"""Iterative RevIN refinement for CPM-masked (horizon) patches, MLX port."""

from __future__ import annotations

import mlx.core as mx

from . import util


def cpm_iterative_revin_refine(
  raw_logits: mx.array,
  revin_n: mx.array,
  revin_mu: mx.array,
  revin_sigma: mx.array,
  patch_cpm_mask: mx.array,
  median_q_idx: int,
  rolls: int,
  patch_len: int,
  num_quantiles: int,
  value_clip: float,
):
  """Refines RevIN stats at CPM-masked patches via iterative estimation.

  For each CPM-masked position the frozen stats (from the last observed patch) are replaced with
  stats that incorporate model-estimated values for all preceding CPM patches in the block.
  """
  b, v, n, _ = raw_logits.shape
  median = raw_logits.reshape(b, v, n, rolls, patch_len, num_quantiles)[
    :, :, :, :, :, median_q_idx
  ]
  carry = (mx.zeros((b, v)), mx.zeros((b, v)), mx.zeros((b, v)))
  anchor = mx.zeros((b, v, rolls, patch_len))
  block_offset = mx.zeros((b,), dtype=mx.int32)
  step_masks = mx.zeros((b, v, patch_len), dtype=mx.bool_)
  ref_mu, ref_sigma = [], []
  for i in range(n):
    is_cpm = patch_cpm_mask[:, i : i + 1]
    onehot = (mx.arange(rolls)[None, :] == block_offset[:, None]).astype(mx.float32)
    predicted_step = (onehot[:, None, :, None] * anchor).sum(axis=2)
    new_n, new_mu, new_sigma = util.update_running_stats(
      *carry, predicted_step, step_masks
    )
    out_n = mx.where(is_cpm, new_n, revin_n[:, :, i])
    out_mu = mx.where(is_cpm, new_mu, revin_mu[:, :, i])
    out_sigma = mx.where(is_cpm, new_sigma, revin_sigma[:, :, i])
    new_block_offset = mx.where(
      is_cpm[:, 0], (block_offset + 1) % rolls, mx.zeros_like(block_offset)
    )
    should_update = (new_block_offset == 0)[:, None, None, None]
    step_pred = mx.clip(
      util.revin(median[:, :, i], out_mu, out_sigma, reverse=True),
      -value_clip,
      value_clip,
    )
    anchor = mx.where(should_update, step_pred, anchor)
    carry = (out_n, out_mu, out_sigma)
    block_offset = new_block_offset
    ref_mu.append(out_mu)
    ref_sigma.append(out_sigma)
  return mx.stack(ref_mu, axis=2), mx.stack(ref_sigma, axis=2)
