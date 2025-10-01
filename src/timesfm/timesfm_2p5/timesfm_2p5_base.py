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

"""TimesFM 2p5 base implementation."""

import dataclasses
from typing import Any, Callable

import numpy as np

from .. import configs

ResidualBlockConfig = configs.ResidualBlockConfig
StackedTransformersConfig = configs.StackedTransformersConfig
TransformerConfig = configs.TransformerConfig
ForecastConfig = configs.ForecastConfig


def strip_leading_nans(arr):
  """Removes contiguous NaN values from the beginning of a NumPy array.

  Args:
    arr: The input NumPy array.

  Returns:
    A new NumPy array with leading NaN values removed.
    If the array is all NaNs or empty, returns an empty array.
  """

  isnan = np.isnan(arr)
  first_valid_index = np.argmax(~isnan)
  return arr[first_valid_index:]


def linear_interpolation(arr):
  """Performs linear interpolation to fill NaN values in a 1D numpy array.

  Args:
      arr: The 1D numpy array containing NaN values.

  Returns:
      A new numpy array with NaN values filled using linear interpolation,
      or the original array if no NaNs are present.
      Returns None if the input is not a 1D array.
      Returns the original array if there are no NaN values.
  """

  nans = np.isnan(arr)
  if not np.any(nans):  # Check if there are any NaNs
    return arr

  def x(z):
    return z.nonzero()[0]

  nans_indices = x(nans)
  non_nans_indices = x(~nans)
  non_nans_values = arr[~nans]

  try:
    arr[nans] = np.interp(nans_indices, non_nans_indices, non_nans_values)
  except ValueError:
    if non_nans_values:
      mu = np.nanmean(arr)
    else:
      mu = 0.0
    arr = np.where(np.isfinite(arr), arr, mu)
  return arr


@dataclasses.dataclass(frozen=True)
class TimesFM_2p5_200M_Definition:
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
      fuse_qkv=True,
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


class TimesFM_2p5:
  """Abstract base class for TimesFM models.

  Attributes:
    forecast_config: Configuration for forecasting flags.
    compiled_decode: Compiled decode function.
    global_batch_size: Global batch size.
  """

  forecast_config: ForecastConfig | None = None
  compiled_decode: Callable[..., Any] | None = None
  global_batch_size: int = 0

  def load_checkpoint(self, path: str):
    """Loads a TimesFM model from a checkpoint."""
    raise NotImplementedError()

  def compile(self, forecast_config: ForecastConfig | None = None):
    """Compiles the TimesFM model for fast decoding."""
    raise NotImplementedError()

  def forecast(
    self, horizon: int, inputs: list[np.ndarray]
  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts the time series."""
    if self.compiled_decode is None:
      raise RuntimeError("Model is not compiled. Please call compile() first.")

    assert self.global_batch_size > 0
    assert self.forecast_config is not None

    context = self.forecast_config.max_context
    num_inputs = len(inputs)
    if (w := num_inputs % self.global_batch_size) != 0:
      inputs += [np.array([0.0] * 3)] * (self.global_batch_size - w)

    output_points = []
    output_quantiles = []
    values = []
    masks = []
    idx = 0
    for each_input in inputs:
      value = linear_interpolation(strip_leading_nans(np.array(each_input)))
      if (w := len(value)) >= context:
        value = value[-context:]
        mask = np.zeros_like(value, dtype=bool)
      else:
        mask = np.array([True] * (context - w) + [False] * w)
        value = np.pad(value, (context - w, 0), "constant", constant_values=0.0)
      values.append(value)
      masks.append(mask)
      idx += 1
      if idx == self.global_batch_size:
        idx = 0
        point_forecast, quantile_forecast = self.compiled_decode(horizon, values, masks)
        output_points.append(point_forecast)
        output_quantiles.append(quantile_forecast)
        values = []
        masks = []

    output_points = np.concatenate(output_points, axis=0)
    output_quantiles = np.concatenate(output_quantiles, axis=0)
    return output_points[:num_inputs], output_quantiles[:num_inputs]
