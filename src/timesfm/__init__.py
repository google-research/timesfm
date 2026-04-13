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

import warnings
from importlib.metadata import PackageNotFoundError, version

from .configs import ForecastConfig

try:
  __version__ = version("timesfm")
except PackageNotFoundError:
  __version__ = "unknown"

try:
  from .timesfm_2p5 import timesfm_2p5_torch
  TimesFM_2p5_200M_torch = timesfm_2p5_torch.TimesFM_2p5_200M_torch
except ImportError as _e:
  if "torch" not in str(_e).lower() and "No module named" not in str(_e):
    warnings.warn(
      f"timesfm[torch] is installed but failed to import: {_e}",
      ImportWarning,
      stacklevel=2,
    )

try:
  from .timesfm_2p5 import timesfm_2p5_flax
  TimesFM_2p5_200M_flax = timesfm_2p5_flax.TimesFM_2p5_200M_flax
except ImportError as _e:
  if "flax" not in str(_e).lower() and "jax" not in str(_e).lower() and "No module named" not in str(_e):
    warnings.warn(
      f"timesfm[flax] is installed but failed to import: {_e}",
      ImportWarning,
      stacklevel=2,
    )

__all__ = ["__version__", "ForecastConfig"]
if "TimesFM_2p5_200M_torch" in dir():
  __all__.append("TimesFM_2p5_200M_torch")
if "TimesFM_2p5_200M_flax" in dir():
  __all__.append("TimesFM_2p5_200M_flax")
