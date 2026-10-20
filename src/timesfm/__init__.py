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

import logging

from .configs import ForecastConfig

_root_handlers = tuple(logging.root.handlers)

try:
  from .timesfm_2p5 import timesfm_2p5_torch
  TimesFM_2p5_200M_torch = timesfm_2p5_torch.TimesFM_2p5_200M_torch
except ImportError:
  pass

try:
  from .timesfm_2p5 import timesfm_2p5_flax
  TimesFM_2p5_200M_flax = timesfm_2p5_flax.TimesFM_2p5_200M_flax
except ImportError:
  pass

try:
  from timesfm3 import TimesFM3Forecaster, TimesFM3Torch
except ImportError:
  pass

def _restore_root_handlers(root_handlers):
  """Removes handlers added as a side effect of optional imports."""
  for handler in tuple(logging.root.handlers):
    if handler not in root_handlers:
      logging.root.removeHandler(handler)


_restore_root_handlers(_root_handlers)
del _restore_root_handlers, _root_handlers
