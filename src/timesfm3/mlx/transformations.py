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

"""Reversible input transformations for the MLX TimesFM3 model.

Port of the PyTorch ``transformations.py``: signed_log, signed_sqrt, identity. Each is a callable
with signature ``(x, reverse=False) -> x'``. TimesFM 3.0 uses ``identity`` by default.
"""

from __future__ import annotations

import mlx.core as mx


def signed_log(x: mx.array, *, reverse: bool = False) -> mx.array:
  """Signed-log transform: ``sign(x) * log(1 + |x|)``."""
  if reverse:
    return mx.sign(x) * mx.expm1(mx.abs(x))
  return mx.sign(x) * mx.log1p(mx.abs(x))


def signed_sqrt(x: mx.array, *, reverse: bool = False) -> mx.array:
  """Signed-sqrt transform: ``sign(x) * sqrt(|x|)``."""
  if reverse:
    return mx.sign(x) * mx.square(x)
  return mx.sign(x) * mx.sqrt(mx.abs(x))


def identity(x: mx.array, *, reverse: bool = False) -> mx.array:
  """Identity (no-op) transform."""
  del reverse
  return x


_REGISTRY = {
  "signed_log": signed_log,
  "signed_sqrt": signed_sqrt,
  "identity": identity,
}


def get_transform(name: str):
  """Looks up a reversible transformation by name."""
  if name not in _REGISTRY:
    raise ValueError(f"Unknown transform name: {name}")
  return _REGISTRY[name]
