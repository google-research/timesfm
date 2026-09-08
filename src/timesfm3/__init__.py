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

"""TimesFM3 API.

The PyTorch backend lives in ``timesfm3.torch`` and the MLX (Apple Silicon) backend in
``timesfm3.mlx``. For backward compatibility the PyTorch API is also re-exported at the top level
(``from timesfm3 import TimesFM3Forecaster``); these names are resolved lazily so that importing
``timesfm3`` — or ``timesfm3.mlx`` — does not require PyTorch to be installed.
"""

_TORCH_EXPORTS = frozenset(
  {
    "ForecastOutput",
    "ModelConfig",
    "ResidualBlockConfig",
    "StackedTransformersConfig",
    "TimesFM3Evaluator",
    "TimesFM3Forecaster",
    "TimesFM3Torch",
    "TransformerConfig",
    "_ModelConfig",
  }
)

__all__ = sorted(_TORCH_EXPORTS)


def __getattr__(name):  # PEP 562: lazily re-export the torch backend at the top level
  if name in _TORCH_EXPORTS:
    from . import torch as _torch_backend

    return getattr(_torch_backend, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
