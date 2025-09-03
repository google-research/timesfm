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

"""TimesFM implementation."""

from typing import Any, Callable
import numpy as np
from . import abstract


class TimesFM_2p5:  # pylint: disable=invalid-name
  """Abstract base class for TimesFM models."""

  config = abstract.TimesFM_2p5_200M()
  forecast_config: abstract.ForecastConfig | None = None
  compiled_decode: Callable[..., Any] | None = None

  def load_from_checkpoint(self, path: str):
    """Loads a TimesFM model from a checkpoint."""
    raise NotImplementedError()

  def compile(self, forecast_config: abstract.ForecastConfig | None = None):
    """Compiles the TimesFM model for fast decoding."""
    raise NotImplementedError()

  def forecast(
      self,
      inputs,
      *,
      options: abstract.ForecastConfig | None = None,
      **kwargs,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts the time series."""
    raise NotImplementedError()
