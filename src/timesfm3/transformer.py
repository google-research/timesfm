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

"""Backward-compatibility shim for ``timesfm3.transformer``.

The PyTorch backend moved to ``timesfm3.torch.transformer``. This re-exports it so that
existing ``from timesfm3.transformer import ...`` imports keep working; new code should
import from ``timesfm3.torch.transformer`` (or the top-level ``timesfm3`` API).
"""

from .torch.transformer import *  # noqa: F401,F403
