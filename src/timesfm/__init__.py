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

"""TimesFM API."""

from .configs import ForecastConfig as ForecastConfig

# Optional backends (PyTorch, Flax) may be absent when the matching extra
# (`timesfm[torch]` / `timesfm[flax]`) is not installed. Keep the original
# import error so that touching such a backend explains what actually went
# wrong instead of raising a bare `AttributeError` (see #318 and #287).
_BACKEND_IMPORT_ERRORS: dict[str, ImportError] = {}

try:
  from .timesfm_2p5 import timesfm_2p5_torch

  TimesFM_2p5_200M_torch = timesfm_2p5_torch.TimesFM_2p5_200M_torch
except ImportError as exc:
  _BACKEND_IMPORT_ERRORS["TimesFM_2p5_200M_torch"] = exc

try:
  from .timesfm_2p5 import timesfm_2p5_flax

  TimesFM_2p5_200M_flax = timesfm_2p5_flax.TimesFM_2p5_200M_flax
except ImportError as exc:
  _BACKEND_IMPORT_ERRORS["TimesFM_2p5_200M_flax"] = exc

try:
  from timesfm3 import TimesFM3Forecaster as TimesFM3Forecaster
  from timesfm3 import TimesFM3Torch as TimesFM3Torch
except ImportError as exc:
  _BACKEND_IMPORT_ERRORS["TimesFM3Forecaster"] = exc
  _BACKEND_IMPORT_ERRORS["TimesFM3Torch"] = exc

_BACKEND_INSTALL_HINTS = {
  "TimesFM_2p5_200M_torch": 'pip install "timesfm[torch]"',
  "TimesFM_2p5_200M_flax": 'pip install "timesfm[flax]"',
  "TimesFM3Forecaster": 'pip install "timesfm[torch]"',
  "TimesFM3Torch": 'pip install "timesfm[torch]"',
}


def __getattr__(name: str) -> object:
  """Surface optional-backend import failures with an actionable message.

  Mirrors the lazy re-export pattern already used by `timesfm3.__getattr__`:
  names whose backend failed to import raise an `ImportError` that chains the
  original failure and names the extra to install, while unknown names keep
  raising the standard `AttributeError` that introspection tools expect.
  """
  if name in _BACKEND_IMPORT_ERRORS:
    original_error = _BACKEND_IMPORT_ERRORS[name]
    raise ImportError(
      f"timesfm.{name} could not be imported ({original_error!r})."
      f" Install the optional backend with: {_BACKEND_INSTALL_HINTS[name]}"
    ) from original_error
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
