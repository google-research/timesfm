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

"""Numeric primitives for the MLX TimesFM3 model (RevIN, running stats, roll, stitch)."""

from __future__ import annotations

import mlx.core as mx

# torch.finfo(float32).eps — matches nn.RMSNorm(eps=None) in the torch backend.
RMS_EPS = 1.1920929e-07
DIV_TOL = 1e-6


def softplus(x: mx.array) -> mx.array:
  return mx.logaddexp(x, mx.zeros_like(x))


def rms_norm(x: mx.array, weight: mx.array | None, eps: float = RMS_EPS) -> mx.array:
  out = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
  return out * weight if weight is not None else out


def roll_back1(x: mx.array, axis: int) -> mx.array:
  """``torch.roll(x, shifts=-1, dims=axis)``: shift each slice one step toward index 0, wrapping."""
  n = x.shape[axis]
  idx = mx.concatenate([mx.arange(1, n), mx.array([0])])
  return mx.take(x, idx, axis=axis)


def revin(
  x: mx.array, mu: mx.array, sigma: mx.array, reverse: bool = False
) -> mx.array:
  """Reversible per-instance normalization; mu/sigma have 1 or 2 fewer dims than x."""
  if mu.ndim == x.ndim - 1:
    mu, sigma = mu[..., None], sigma[..., None]
  elif mu.ndim == x.ndim - 2:
    mu, sigma = mu[..., None, None], sigma[..., None, None]
  if reverse:
    return x * sigma + mu
  safe_sigma = mx.where(sigma < DIV_TOL, 1.0, sigma)
  return (x - mu) / safe_sigma


def update_running_stats(n, mu, sigma, x, mask):
  """Welford-style merge of a patch ``(x, mask)`` into running ``(n, mu, sigma)``."""
  legit = ~mask
  inc_n = legit.astype(mx.float32).sum(axis=-1)
  safe_inc_n = mx.where(inc_n == 0, 1.0, inc_n)
  inc_sum = mx.where(legit, x, 0.0).sum(axis=-1)
  inc_mu = mx.where(inc_n == 0, 0.0, inc_sum / safe_inc_n)
  diff_sq = mx.where(legit, (x - inc_mu[..., None]) ** 2, 0.0)
  inc_var = mx.where(inc_n == 0, 0.0, diff_sq.sum(axis=-1) / safe_inc_n)
  inc_sigma = mx.sqrt(inc_var)
  new_n = n + inc_n
  safe_new_n = mx.where(new_n == 0, 1.0, new_n)
  new_mu = mx.where(new_n == 0, 0.0, (n * mu + inc_mu * inc_n) / safe_new_n)
  new_var = mx.where(
    new_n == 0,
    0.0,
    (
      n * sigma * sigma
      + inc_n * inc_sigma * inc_sigma
      + n * (mu - new_mu) ** 2
      + inc_n * (inc_mu - new_mu) ** 2
    )
    / safe_new_n,
  )
  return new_n, new_mu, mx.sqrt(new_var)


def get_running_stats(values: mx.array, masks: mx.array):
  """Cumulative causal running ``(n, mean, std)`` per patch. Shapes ``(b, v, n, p)``."""
  b, v, n, _ = values.shape
  cur = (mx.zeros((b, v)), mx.zeros((b, v)), mx.zeros((b, v)))
  out_n, out_mu, out_sigma = [], [], []
  for i in range(n):
    cur = update_running_stats(*cur, values[:, :, i, :], masks[:, :, i, :])
    out_n.append(cur[0])
    out_mu.append(cur[1])
    out_sigma.append(cur[2])
  return mx.stack(out_n, axis=2), mx.stack(out_mu, axis=2), mx.stack(out_sigma, axis=2)


def output_patch_via_roll(x: mx.array, rolls: int):
  """Build ``(b, v, n, p*rolls)`` future-covariate patches by rolling the patch index."""
  b, v, n, p = x.shape
  cur = x
  parts = []
  for _ in range(rolls):
    cur = roll_back1(cur, axis=2)
    parts.append(cur)
  result = mx.concatenate(parts, axis=-1)
  patch_idx = mx.arange(n)[:, None]
  point_idx = mx.arange(rolls * p)[None, :]
  source_patch = patch_idx + 1 + point_idx // p
  wrap = (source_patch >= n)[None, None, :, :]
  return result, wrap


def stitch_patches(patch_preds: mx.array, patch_len: int) -> mx.array:
  """Linearly stitch overlapping patch predictions. ``(b, v, np, patch+overlap, q)``."""
  b, v, num_patches, total_len, q = patch_preds.shape
  overlap = total_len - patch_len
  if num_patches == 1:
    return patch_preds[:, :, 0, :, :]
  w = mx.linspace(1.0, 0.0, overlap).reshape(1, 1, 1, overlap, 1)
  first = patch_preds[:, :, 0, :patch_len, :]
  prev, nxt = patch_preds[:, :, :-1, :, :], patch_preds[:, :, 1:, :, :]
  stitched = w * prev[:, :, :, patch_len:, :] + (1.0 - w) * nxt[:, :, :, :overlap, :]
  middles = nxt[:, :, :, overlap:patch_len, :]
  chunks = mx.concatenate([stitched, middles], axis=3).reshape(
    b, v, (num_patches - 1) * patch_len, q
  )
  tail = patch_preds[:, :, -1, patch_len:, :]
  return mx.concatenate([first, chunks, tail], axis=2)
