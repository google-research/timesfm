"""Reversible transformations for time-series models (PyTorch).

Direct port of the Flax transformations.py: signed_log, signed_sqrt, identity.
Each transformation is a callable with signature (x, reverse=False) -> x'.
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch


class TransformFn(Protocol):
  """Protocol for a reversible transformation function."""

  def __call__(self, x: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
    ...


def signed_log(x: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
  """Signed-log transform: sign(x) * log(1 + |x|)."""
  if reverse:
    return torch.sign(x) * torch.expm1(torch.abs(x))
  return torch.sign(x) * torch.log1p(torch.abs(x))


def _max_output_signed_log(value_clip: float) -> torch.Tensor:
  return torch.log1p(torch.tensor(value_clip))


def signed_sqrt(x: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
  """Signed-sqrt transform: sign(x) * sqrt(|x|)."""
  if reverse:
    return torch.sign(x) * torch.square(x)
  return torch.sign(x) * torch.sqrt(torch.abs(x))


def _max_output_signed_sqrt(value_clip: float) -> torch.Tensor:
  return torch.sqrt(torch.tensor(value_clip))


def identity(x: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
  """Identity (no-op) transform."""
  del reverse
  return x


def _max_output_identity(value_clip: float) -> torch.Tensor:
  return torch.tensor(value_clip, dtype=torch.float32)


_REGISTRY: dict[
    str,
    tuple[TransformFn, Callable[[float], torch.Tensor]],
] = {
    "signed_log": (signed_log, _max_output_signed_log),
    "signed_sqrt": (signed_sqrt, _max_output_signed_sqrt),
    "identity": (identity, _max_output_identity),
}


def get_transform(name: str) -> TransformFn:
  """Looks up a reversible transformation by name."""
  if name not in _REGISTRY:
    raise ValueError(f"Unknown transform name: {name}")
  return _REGISTRY[name][0]


def max_output(name: str, value_clip: float) -> torch.Tensor:
  """Returns the maximum output magnitude for a transform given a value clip."""
  if name not in _REGISTRY:
    raise ValueError(f"Unknown transform name: {name}")
  return _REGISTRY[name][1](value_clip)
