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

"""Backward-compatibility shim for ``timesfm3.timesfm3_forecaster``.

The PyTorch backend moved to ``timesfm3.torch.timesfm3_forecaster``. This module is an alias
of it so that existing ``from timesfm3.timesfm3_forecaster import ...`` imports keep working;
new code should import from ``timesfm3.torch.timesfm3_forecaster`` (or the top-level
``timesfm3`` API).
"""

import sys

from .torch import timesfm3_forecaster as _timesfm3_forecaster

# Alias the module rather than star-importing it: `import *` skips names with a
# leading underscore (`_ModelConfig`, `_make_torch_model`, ...), and a copied
# name cannot be patched, because the backend keeps reading its own global.
sys.modules[__name__] = _timesfm3_forecaster
