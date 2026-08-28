"""Normalization layers for TimesFM3 PyTorch."""

from __future__ import annotations

import math

import torch
from torch import nn

_RECIPROCAL_OF_SOFTPLUS_0 = 1.442695041


class PerDimScale(nn.Module):
  """Per-dimension scaling (Pax-style).

  Replaces the standard 1/sqrt(d) query scaling with a learnable:
    x * RECIPROCAL_OF_SOFTPLUS_0 / sqrt(num_dims) * softplus(per_dim_scale)

  The per_dim_scale parameter is initialized to zeros, so at init time
  softplus(0) ≈ 0.693..., and the net scale is close to 1/sqrt(d).
  """

  def __init__(self, num_dims: int):
    super().__init__()
    self.num_dims = num_dims
    self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Applies per-dim scaling to the last dimension of x."""
    return (
      x
      * _RECIPROCAL_OF_SOFTPLUS_0
      / math.sqrt(self.num_dims)
      * torch.nn.functional.softplus(self.per_dim_scale)
    )
