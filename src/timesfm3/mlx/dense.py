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


class ResidualBlock(nn.Module):
  """Two linear layers with a linear residual connection (no bias, no prenorm).

  ``output_layer(relu(hidden_layer(x))) + residual_layer(x)`` -- matches the pre-transformer
  residual block of the PyTorch backend.
  """

  def __init__(self, in_dim: int, out_dim: int):
    super().__init__()
    self.hidden_layer = nn.Linear(in_dim, out_dim, bias=False)
    self.output_layer = nn.Linear(out_dim, out_dim, bias=False)
    self.residual_layer = nn.Linear(in_dim, out_dim, bias=False)

  def __call__(self, x: mx.array) -> mx.array:
    return self.output_layer(nn.relu(self.hidden_layer(x))) + self.residual_layer(x)
