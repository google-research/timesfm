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

"""Abstract configs for TimesFM-3 layers."""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class ResidualBlockConfig:
  """Framework-agnostic config for a residual block."""

  hidden_dims: int
  output_dims: int
  use_bias: bool
  activation: Literal["relu", "swish", "none"]
  dropout: float = 0.0
  identity_skip: bool = False
  prenorm: Literal["rms", "none"] = "none"


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Framework-agnostic config for a transformer."""

  model_dims: int
  hidden_dims: int
  num_heads: int
  attention_norm: Literal["rms"]
  feedforward_norm: Literal["rms"]
  qk_norm: Literal["rms", "none"]
  use_bias: bool
  use_rope_seq: bool
  use_rope_var: bool
  ff_activation: Literal["relu", "swish", "none"]
  deterministic: bool
  v_norm: Literal["rms", "none"] = "none"
  causal_attention: bool = True
  debug_no_masking: bool = False
  training: bool = True
  use_memory_efficient_attention: bool = True
  paired_token_skip_second: bool = False
  max_variates: int = 32
  # PyTorch-only: when True uses F.scaled_dot_product_attention.
  use_sdpa: bool = True


@dataclasses.dataclass(frozen=True)
class StackedTransformersConfig:
  """Framework-agnostic config for a stacked transformers."""

  num_layers: int
  transformer: TransformerConfig
  use_remat: bool = True
