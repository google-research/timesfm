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

"""Dense layers for TimesFM."""

import torch
from torch import nn

from .. import configs


class ResidualBlock(nn.Module):
  """Residual block with two linear layers and a linear residual connection."""

  def __init__(self, config: configs.ResidualBlockConfig):
    super().__init__()
    self.config = config
    self.hidden_layer = nn.Linear(
        in_features=config.input_dims,
        out_features=config.hidden_dims,
        bias=config.use_bias,
    )
    self.output_layer = nn.Linear(
        in_features=config.hidden_dims,
        out_features=config.output_dims,
        bias=config.use_bias,
    )
    self.residual_layer = nn.Linear(
        in_features=config.input_dims,
        out_features=config.output_dims,
        bias=config.use_bias,
    )
    if config.activation == "relu":
      self.activation = nn.ReLU()
    elif config.activation == "swish":
      self.activation = nn.SiLU()
    elif config.activation == "none":
      self.activation = nn.Identity()
    else:
      raise ValueError(f"Activation: {config.activation} not supported.")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.output_layer(
        self.activation(self.hidden_layer(x))
    ) + self.residual_layer(x)


class RandomFourierFeatures(nn.Module):
  """Random Fourier features layer."""

  def __init__(self, config: configs.RandomFourierFeaturesConfig):
    super().__init__()
    self.config = config

    if config.output_dims % 4 != 0:
      raise ValueError(
          f"Output dims must be a multiple of 4: {config.output_dims} % 4 != 0."
      )
    num_projected_features = config.output_dims // 4

    self.phase_shifts = nn.Parameter(torch.zeros(2, num_projected_features))
    self.projection_layer = nn.Linear(
        in_features=config.input_dims,
        out_features=num_projected_features,
        bias=config.use_bias,
    )
    self.residual_layer = nn.Linear(
        in_features=config.input_dims,
        out_features=config.output_dims,
        bias=config.use_bias,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    projected = self.projection_layer(x)
    cos_features = torch.cos(projected)
    sin_features = torch.sin(projected)
    sq_wave_1 = torch.sign(torch.sin(projected + self.phase_shifts[0, :]))
    sq_wave_2 = torch.sign(torch.sin(projected + self.phase_shifts[1, :]))
    fourier_features = torch.cat(
        [cos_features, sin_features, sq_wave_1, sq_wave_2], dim=-1
    )
    residual = self.residual_layer(x)
    return fourier_features + residual
