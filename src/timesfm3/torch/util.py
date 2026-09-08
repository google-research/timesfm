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

"""Utility functions and classes for TimesFM3 PyTorch implementation."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F
from safetensors import torch as safetensors_torch

_TOLERANCE = 1e-6


def _make_safe_for_division(values: torch.Tensor) -> torch.Tensor:
  """Handles near zero values."""
  return torch.where(values < _TOLERANCE, 1.0, values)


@dataclasses.dataclass
class DecodeCache:
  """Cache for autoregressive decoding.

  Attributes:
    next_index: The next index to decode for each batch element.
      Shape: (batch_leading,).
    num_front_masked: Number of front masked tokens for each batch element.
      Shape: (batch_leading,).
    key: The key cache. Shape: (batch_leading, cache_len, num_heads, head_dim).
    value: The value cache. Same shape as key.
  """

  next_index: torch.Tensor
  num_front_masked: torch.Tensor
  key: torch.Tensor
  value: torch.Tensor

  @classmethod
  def init_decode_cache(
    cls,
    num_layers: int,
    batch_size: int,
    num_variates: int,
    num_total_input_patches: int,
    num_heads: int,
    head_dim: int,
    device: torch.device | None = None,
  ) -> list[DecodeCache]:
    """Initializes a list of decode caches for stacked layers.

    Args:
      num_layers: The number of transformer layers.
      batch_size: The batch size.
      num_variates: The number of variates.
      num_total_input_patches: Total number of patches the cache should hold.
      num_heads: The number of attention heads.
      head_dim: The head dimension.
      device: The device to create tensors on.

    Returns:
      A list of DecodeCache, one per layer.
    """
    leading_size = batch_size * num_variates
    return [
      cls(
        next_index=torch.zeros(leading_size, dtype=torch.int32, device=device),
        num_front_masked=torch.zeros(leading_size, dtype=torch.int32, device=device),
        key=torch.zeros(
          leading_size,
          num_total_input_patches,
          num_heads,
          head_dim,
          device=device,
        ),
        value=torch.zeros(
          leading_size,
          num_total_input_patches,
          num_heads,
          head_dim,
          device=device,
        ),
      )
      for _ in range(num_layers)
    ]


def update_running_stats(
  n: torch.Tensor,
  mu: torch.Tensor,
  sigma: torch.Tensor,
  x: torch.Tensor,
  mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Updates running stats with a new patch of data.

  Args:
    n: Count of seen non-masked elements. Shape: (b, v).
    mu: Running mean. Shape: (b, v).
    sigma: Running std. Shape: (b, v).
    x: New data patch. Shape: (b, v, p).
    mask: Boolean mask where True = masked/invalid. Shape: (b, v, p).

  Returns:
    Tuple of (new_n, new_mu, new_sigma), each of shape (b, v).
  """
  is_legit = ~mask
  is_legit_f = is_legit.float()
  inc_n = is_legit_f.sum(dim=-1)

  # mean of valid elements in patch
  x_masked = torch.where(is_legit, x, torch.zeros_like(x))
  inc_sum = x_masked.sum(dim=-1)
  inc_mu = torch.where(inc_n == 0, torch.zeros_like(inc_sum), inc_sum / inc_n)

  # std of valid elements in patch
  x_diff_sq = torch.where(
    is_legit, (x - inc_mu.unsqueeze(-1)) ** 2, torch.zeros_like(x)
  )
  inc_var = torch.where(
    inc_n == 0,
    torch.zeros_like(inc_sum),
    x_diff_sq.sum(dim=-1) / inc_n,
  )
  inc_sigma = torch.sqrt(inc_var)

  new_n = n + inc_n
  new_mu = torch.where(
    new_n == 0,
    torch.zeros_like(mu),
    (n * mu + inc_mu * inc_n) / new_n,
  )
  new_sigma = torch.sqrt(
    torch.where(
      new_n == 0,
      torch.zeros_like(sigma),
      (
        n * sigma * sigma
        + inc_n * inc_sigma * inc_sigma
        + n * (mu - new_mu) * (mu - new_mu)
        + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)
      )
      / new_n,
    )
  )
  return new_n, new_mu, new_sigma


def get_running_stats(
  values: torch.Tensor,
  masks: torch.Tensor,
  *,
  segment_ids: torch.Tensor | None = None,
  initial_stats: (tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None) = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Computes cumulative running statistics patch-by-patch.

  For each variate, patch `i` gets stats computed from all unmasked values
  in patches 0 through i (inclusive). Statistics reset at segment boundaries.

  Args:
    values: Input values. Shape: (b, v, n, p).
    masks: Boolean mask (True=masked). Shape: (b, v, n, p).
    segment_ids: Segment IDs. Shape: (b, n). If None, no segment resets.
    initial_stats: Optional initial (n, mu, sigma), each shape (b, v).

  Returns:
    Tuple of (running_n, running_mu, running_sigma), each shape (b, v, n).
  """
  b, v, n, _ = values.shape
  device = values.device

  if initial_stats is None:
    init_n = torch.zeros((b, v), dtype=torch.float32, device=device)
    init_mu = torch.zeros((b, v), dtype=torch.float32, device=device)
    init_sigma = torch.zeros((b, v), dtype=torch.float32, device=device)
  else:
    init_n, init_mu, init_sigma = initial_stats

  # Determine segment reset points
  if segment_ids is None:
    is_new_segment = torch.zeros((b, n), dtype=torch.bool, device=device)
  else:
    shifted = F.pad(segment_ids[:, :-1], (1, 0), value=-1)
    is_new_segment = segment_ids != shifted

  all_n = []
  all_mu = []
  all_sigma = []
  cur_n, cur_mu, cur_sigma = init_n, init_mu, init_sigma

  for i in range(n):
    # Reset stats at new segments
    reset = is_new_segment[:, i].unsqueeze(-1)  # (b, 1)  # pyrefly: ignore[bad-index]
    cur_n = torch.where(reset, init_n, cur_n)
    cur_mu = torch.where(reset, init_mu, cur_mu)
    cur_sigma = torch.where(reset, init_sigma, cur_sigma)

    cur_n, cur_mu, cur_sigma = update_running_stats(
      cur_n, cur_mu, cur_sigma, values[:, :, i, :], masks[:, :, i, :]
    )
    all_n.append(cur_n)
    all_mu.append(cur_mu)
    all_sigma.append(cur_sigma)

  return (
    torch.stack(all_n, dim=2),
    torch.stack(all_mu, dim=2),
    torch.stack(all_sigma, dim=2),
  )


def revin(
  x: torch.Tensor,
  mu: torch.Tensor,
  sigma: torch.Tensor,
  reverse: bool = False,
) -> torch.Tensor:
  """Reversible per-instance normalization.

  Automatically expands mu/sigma dims to match x.

  Args:
    x: Input tensor. Shape: (b, ..., d).
    mu: Mean tensor. Shape: (b, ...) with 1 or 2 fewer dims than x.
    sigma: Std tensor. Same shape as mu.
    reverse: If True, applies reverse normalization (denormalize).

  Returns:
    Normalized or denormalized tensor, same shape as x.
  """
  if mu.dim() == x.dim() - 1:
    mu = mu.unsqueeze(-1)
    sigma = sigma.unsqueeze(-1)
  elif mu.dim() == x.dim() - 2:
    mu = mu.unsqueeze(-1).unsqueeze(-1)
    sigma = sigma.unsqueeze(-1).unsqueeze(-1)
  else:
    raise ValueError(f"Unsupported shapes for x and mu: {x.shape}, {mu.shape}.")
  if reverse:
    return x * sigma + mu
  else:
    return (x - mu) / _make_safe_for_division(sigma)


def get_output_patch_via_roll(
  x: torch.Tensor, rolls: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Creates labels of output_patch length by rolling the patched inputs.

  Takes patched input (b, v, n, p) and creates output patches of length
  p*rolls by concatenating shifted views of the patches.

  Args:
    x: Patched inputs. Shape: (b, v, n, p).
    rolls: Number of rolls (= output_patch_len / patch_len).

  Returns:
    Tuple of:
      - Rolled output. Shape: (b, v, n, p * rolls).
      - Wrap-around mask. Shape: (1, 1, n, p * rolls) bool.
  """
  b, v, n, p = x.shape
  device = x.device
  rolling_mat = torch.zeros(b, v, n, rolls + 1, p, device=device, dtype=x.dtype)
  rolling_mat[:, :, :, 0, :] = x

  for i in range(rolls):
    rolling_mat[:, :, :, i + 1, :] = torch.roll(
      rolling_mat[:, :, :, i, :], shifts=-1, dims=2
    )

  # Take [1:] along the roll axis and flatten
  result = rolling_mat[:, :, :, 1:, :].reshape(b, v, n, rolls * p)

  # Build wrap-around mask
  patch_idx = torch.arange(n, device=device)
  point_idx = torch.arange(rolls * p, device=device)
  source_patch = patch_idx[:, None] + 1 + point_idx[None, :] // p
  wrap_mask = (source_patch >= n).unsqueeze(0).unsqueeze(0)

  return result, wrap_mask


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
  "relu": F.relu,
  "swish": F.silu,
  "silu": F.silu,
  "none": lambda x: x,
}


def get_activation_fn(
  activation_name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
  """Returns the activation function for the given name."""
  try:
    return _ACTIVATIONS[activation_name]
  except KeyError:
    raise ValueError(
      f"Activation: {activation_name} not supported. Supported "
      f"activations: {list(_ACTIVATIONS.keys())}"
    ) from None


def load_safetensors(
  pytorch_safetensors_path: str,
  device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
  """Loads a PyTorch state dict from a safetensors file.

  Args:
    pytorch_safetensors_path: Path to the safetensors file.
    device: The device to load the tensors onto.

  Returns:
    A dictionary of PyTorch state dict weights.
  """
  expanded_path = os.path.expanduser(pytorch_safetensors_path)
  return safetensors_torch.load_file(expanded_path, device=str(device))


def stitch_patches(
  patch_preds: torch.Tensor,
  patch_len: int,
) -> torch.Tensor:
  """Stitches overlapping patch predictions.

  Each patch predicts patch_len + overlap timepoints, where
  overlap = patch_preds.shape[3] - patch_len is inferred from the input.
  Consecutive patches share overlap timepoints, which are linearly stitched.

  Args:
    patch_preds: Predictions of shape (batch, variates, num_patches, patch_len
      + overlap, num_quantiles).
    patch_len: The patch length.

  Returns:
    Stitched predictions of shape
    (batch, variates, num_patches * patch_len + overlap, num_quantiles).
  """
  b, v, num_patches, total_len, q = patch_preds.shape
  overlap = total_len - patch_len

  if num_patches == 1:
    return patch_preds[:, :, 0, :, :]

  stitch_weights = torch.linspace(
    1.0, 0.0, overlap, device=patch_preds.device, dtype=patch_preds.dtype
  )
  stitch_weights = stitch_weights[None, None, None, :, None]

  first_chunk = patch_preds[:, :, 0, :patch_len, :]

  prev_patches = patch_preds[:, :, :-1, :, :]
  next_patches = patch_preds[:, :, 1:, :, :]

  prev_overlaps = prev_patches[:, :, :, patch_len:, :]
  next_overlaps = next_patches[:, :, :, :overlap, :]

  stitched_overlaps = (
    stitch_weights * prev_overlaps + (1.0 - stitch_weights) * next_overlaps
  )

  middles = next_patches[:, :, :, overlap:patch_len, :]

  output_chunks = torch.cat([stitched_overlaps, middles], dim=3)

  mid = output_chunks.reshape(b, v, (num_patches - 1) * patch_len, q)

  tail = patch_preds[:, :, -1, patch_len:, :]

  return torch.cat([first_chunk, mid, tail], dim=2)
