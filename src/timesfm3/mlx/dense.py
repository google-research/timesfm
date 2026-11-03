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

"""Dense layers for the MLX TimesFM3 model."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from . import normalization

_ACTIVATIONS = {
  "relu": nn.relu,
  "swish": nn.silu,
  "silu": nn.silu,
  "none": lambda x: x,
}


class ResidualBlock(nn.Module):
  """Two linear layers with a residual connection (no bias) -- matches the pre-transformer
  residual block of the PyTorch backend.

  ``output_layer(activation(hidden_layer(prenorm(x)))) + residual_layer(x)`` (or ``+ x`` when
  ``identity_skip``).
  """

  def __init__(
    self,
    in_dim: int,
    out_dim: int,
    activation: str = "relu",
    prenorm: str = "none",
    identity_skip: bool = False,
  ):
    super().__init__()
    self.hidden_layer = nn.Linear(in_dim, out_dim, bias=False)
    self.output_layer = nn.Linear(out_dim, out_dim, bias=False)
    self.identity_skip = identity_skip
    if not identity_skip:
      self.residual_layer = nn.Linear(in_dim, out_dim, bias=False)
    self.activation = _ACTIVATIONS[activation]
    self.pre_norm = normalization.RMSNorm(in_dim) if prenorm == "rms" else None

  def __call__(self, x: mx.array) -> mx.array:
    hidden_input = self.pre_norm(x) if self.pre_norm is not None else x
    hidden = self.activation(self.hidden_layer(hidden_input))
    if self.identity_skip:
      return self.output_layer(hidden) + x
    return self.output_layer(hidden) + self.residual_layer(x)
