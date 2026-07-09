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

"""Normalization layers for TimesFM."""

import torch
from torch import nn


class RMSNorm(nn.Module):
  """RMS normalization."""

  def __init__(
      self,
      num_features: int,
      *,
      epsilon: float = 1e-6,
  ):
    super().__init__()
    self.scale = nn.Parameter(torch.zeros(num_features))
    self.num_features = num_features
    self.epsilon = epsilon

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    var = torch.mean(torch.square(inputs), dim=-1, keepdim=True)
    normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
    normed_inputs = normed_inputs * self.scale
    return normed_inputs
