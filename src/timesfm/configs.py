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


@dataclasses.dataclass(frozen=True)
class ForecastConfig:
  """Options for forecasting.

  Attributes:
    max_context: The maximum context length. This is used by the complied decode
      function at inference time during batched inference. Any input time series
      with length less than max_context will be padded with zeros, and with
      length greater than max_context will be truncated.
    max_horizon: The maximum horizon length. This is used by the complied decode
      function at inference time during batched inference. The compiled cached
      decoding function will by default forecast till max_horizon.
    normalize_inputs: Whether to normalize the inputs. This is useful when the
      raw inputs are of extremely large or small magnitudes which may result in
      numerical issues.
    window_size: The window size for decomposed forecasting.
      TODO(siriuz42):implement it.
    per_core_batch_size: The batch size per core. Used at inference time during
      batched inference when multiple GPU / TPU devices are used.
    use_continuous_quantile_head: Whether to use a separate continuous quantile
      head to avoid quantile collapsing.
    force_flip_invariance: Whether to force flip invariance. TimesFM guarantees
      that TimesFM(aX + b) = a * TimesFM(x) + b for a >= 0 by default. This flag
      extends it to a < 0 as well.
    infer_is_positive: Whether to guarantee nonnegativity of the output if the
      input is nonnegative.
    fix_quantile_crossing: Whether to fix quantile crossing.
    return_backcast: Whether to return backcast.
  """

  max_context: int = 0
  max_horizon: int = 0
  normalize_inputs: bool = False
  window_size: int = 0
  per_core_batch_size: int = 1
  use_continuous_quantile_head: bool = False
  force_flip_invariance: bool = True
  infer_is_positive: bool = True
  fix_quantile_crossing: bool = False
  return_backcast: bool = False


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
  fuse_qkv: bool


@dataclasses.dataclass(frozen=True)
class StackedTransformersConfig:
  """Framework-agnostic config for a stacked transformers."""

  num_layers: int
  transformer: TransformerConfig
