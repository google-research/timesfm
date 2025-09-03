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

"""Abstract configs for TimesFM layers."""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=False)
class ForecastConfig:
  """Options for forecasting."""

  max_context: int = 0
  max_horizon: int = 0
  normalize_inputs: bool = False
  window_size: int = 0
  per_core_batch_size: int = 1
  use_continuous_quantile_head: bool = False
  infer_is_positive: bool = True
  return_forecast_on_context: bool = False


@dataclasses.dataclass(frozen=True)
class ResidualBlockConfig:
  """Framework-agnostic config for a residual block."""

  input_dims: int
  hidden_dims: int
  output_dims: int
  use_bias: bool
  activation: Literal["relu", "swish", "none"]


@dataclasses.dataclass(frozen=True)
class RandomFourierFeaturesConfig:
  """Framework-agnostic config for random fourier features."""

  input_dims: int
  output_dims: int
  projection_stddev: float
  use_bias: bool


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
  use_rotary_position_embeddings: bool
  ff_activation: Literal["relu", "swish", "none"]


@dataclasses.dataclass(frozen=True)
class StackedTransformersConfig:
  """Framework-agnostic config for a stacked transformers."""

  num_layers: int
  transformer: TransformerConfig


@dataclasses.dataclass(frozen=True)
class TimesFM_2p5_200M:  # pylint: disable=invalid-name
  """Framework-agnostic config of TimesFM 2.5."""

  context_limit = 16384
  input_patch_len: int = 32
  output_patch_len: int = 128
  output_quantile_len: int = 1024
  quantiles: list[float] = dataclasses.field(
      default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  )
  decode_index: int = 5
  tokenizer: ResidualBlockConfig = ResidualBlockConfig(
      input_dims=64,
      hidden_dims=1280,
      output_dims=1280,
      use_bias=True,
      activation="swish",
  )
  stacked_transformers: StackedTransformersConfig = StackedTransformersConfig(
      num_layers=20,
      transformer=TransformerConfig(
          model_dims=1280,
          hidden_dims=1280,
          num_heads=16,
          attention_norm="rms",
          feedforward_norm="rms",
          qk_norm="rms",
          use_bias=False,
          use_rotary_position_embeddings=True,
          ff_activation="swish",
      ),
  )
  output_projection_point: ResidualBlockConfig = ResidualBlockConfig(
      input_dims=1280,
      hidden_dims=1280,
      output_dims=1280,
      use_bias=False,
      activation="swish",
  )
  output_projection_quantiles: ResidualBlockConfig = ResidualBlockConfig(
      input_dims=1280,
      hidden_dims=1280,
      output_dims=10240,
      use_bias=False,
      activation="swish",
  )
