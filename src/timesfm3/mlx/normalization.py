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

"""Normalization layers for the MLX TimesFM3 model (RMSNorm, PerDimScale)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from . import util

_RECIPROCAL_OF_SOFTPLUS_0 = 1.442695041


class RMSNorm(nn.Module):
  """RMS normalization with a learned per-feature scale."""

  def __init__(self, dims: int):
    super().__init__()
    self.weight = mx.ones((dims,))

  def __call__(self, x: mx.array) -> mx.array:
    return util.rms_norm(x, self.weight)


class PerDimScale(nn.Module):
  """Per-dimension query scaling (Pax-style).

  Replaces the standard 1/sqrt(d) query scaling with a learnable
  ``x * RECIPROCAL_OF_SOFTPLUS_0 / sqrt(num_dims) * softplus(per_dim_scale)``.
  """

  def __init__(self, num_dims: int):
    super().__init__()
    self.per_dim_scale = mx.zeros((num_dims,))
    self._num_dims = num_dims

  def __call__(self, x: mx.array) -> mx.array:
    scale = _RECIPROCAL_OF_SOFTPLUS_0 / math.sqrt(self._num_dims)
    return x * scale * util.softplus(self.per_dim_scale)
