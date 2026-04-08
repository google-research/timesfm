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

"""LoRA and DoRA adapter layers for PyTorch, plus injection / merging helpers.

References:
  LoRA — https://arxiv.org/abs/2106.09685
  DoRA — https://arxiv.org/abs/2402.09353
"""

import math
import os
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .config import PEFTConfig


# ---------------------------------------------------------------------------
# Adapter layers
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
  """Drop-in replacement for ``nn.Linear`` that adds a low-rank branch.

  ``output = base_linear(x) + (dropout(x) @ A @ B) * (alpha / rank)``

  *A* is Kaiming-uniform initialised; *B* is zero-initialised so the
  effective delta is zero at init and the pretrained model is preserved.
  """

  def __init__(
    self,
    base_linear: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
  ):
    super().__init__()
    self.base_linear = base_linear
    self.rank = rank
    self.scaling = alpha / rank

    in_f = base_linear.in_features
    out_f = base_linear.out_features

    self.lora_A = nn.Parameter(torch.empty(in_f, rank))
    self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # Freeze the pretrained weight.
    self.base_linear.weight.requires_grad = False
    if self.base_linear.bias is not None:
      self.base_linear.bias.requires_grad = False

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    base_out = self.base_linear(x)
    lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
    return base_out + lora_out

  def merge_weights(self) -> nn.Linear:
    """Fold the LoRA delta into the base ``nn.Linear`` and return it."""
    with torch.no_grad():
      delta = (self.lora_A @ self.lora_B * self.scaling).T  # (out, in)
      self.base_linear.weight.add_(delta)
    return self.base_linear


class DoRALinear(nn.Module):
  """Weight-Decomposed Low-Rank Adaptation (DoRA).

  Decomposes the adapted weight into *magnitude* and *direction*::

      W' = m · (W + ΔW) / ‖W + ΔW‖_col

  ``m`` is initialised from the pretrained column norms so the model
  starts at the same operating point.
  """

  def __init__(
    self,
    base_linear: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
  ):
    super().__init__()
    self.base_linear = base_linear
    self.rank = rank
    self.scaling = alpha / rank

    in_f = base_linear.in_features
    out_f = base_linear.out_features

    self.lora_A = nn.Parameter(torch.empty(in_f, rank))
    self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    # Magnitude vector — initialised from pretrained column norms.
    with torch.no_grad():
      col_norms = base_linear.weight.norm(dim=1)
    self.magnitude = nn.Parameter(col_norms.clone())

    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    self.base_linear.weight.requires_grad = False
    if self.base_linear.bias is not None:
      self.base_linear.bias.requires_grad = False

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    delta_W = (self.lora_A @ self.lora_B * self.scaling).T  # (out, in)
    adapted_W = self.base_linear.weight + delta_W
    col_norm = adapted_W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_prime = self.magnitude.unsqueeze(1) * (adapted_W / col_norm)
    return F.linear(x, W_prime, self.base_linear.bias)

  def merge_weights(self) -> nn.Linear:
    """Fold DoRA into the base ``nn.Linear`` and return it."""
    with torch.no_grad():
      delta_W = (self.lora_A @ self.lora_B * self.scaling).T
      adapted_W = self.base_linear.weight + delta_W
      col_norm = adapted_W.norm(dim=1, keepdim=True).clamp(min=1e-8)
      self.base_linear.weight.copy_(
        self.magnitude.unsqueeze(1) * (adapted_W / col_norm)
      )
    return self.base_linear


# ---------------------------------------------------------------------------
# Injection / merge helpers
# ---------------------------------------------------------------------------

_ADAPTER_CLS = {"lora": LoRALinear, "dora": DoRALinear}


def inject_adapters(
  model: nn.Module,
  config: PEFTConfig,
) -> nn.Module:
  """Inject LoRA / DoRA adapters into a ``TimesFM_2p5_200M_torch_module``.

  All base parameters are frozen.  Only adapter parameters (and, optionally,
  the output-projection heads) remain trainable.

  Args:
    model: The ``TimesFM_2p5_200M_torch_module`` instance.
    config: PEFT configuration.

  Returns:
    The same model, mutated in-place with adapter wrappers.
  """
  adapter_cls = _ADAPTER_CLS[config.adapter_type]
  kwargs = dict(rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)
  target = config.target_modules

  # 1. Freeze everything.
  for p in model.parameters():
    p.requires_grad = False

  # 2. Determine which layers get adapters.
  total_layers = model.x  # 20
  if config.num_adapter_layers > 0:
    first_adapter_layer = total_layers - config.num_adapter_layers
  else:
    first_adapter_layer = 0

  # 3. Wrap target nn.Linear modules with adapters.
  for layer_idx in range(total_layers):
    if layer_idx < first_adapter_layer:
      continue
    xf = model.stacked_xf[layer_idx]

    if target in ("all", "attention"):
      # Fused QKV projection (TimesFM 2.5 always uses fuse_qkv=True).
      if hasattr(xf.attn, "qkv_proj") and isinstance(xf.attn.qkv_proj, nn.Linear):
        xf.attn.qkv_proj = adapter_cls(xf.attn.qkv_proj, **kwargs)
      else:
        # Fallback for non-fused Q / K / V.
        for attr in ("query", "key", "value"):
          orig = getattr(xf.attn, attr, None)
          if isinstance(orig, nn.Linear):
            setattr(xf.attn, attr, adapter_cls(orig, **kwargs))
      # Output projection.
      if isinstance(xf.attn.out, nn.Linear):
        xf.attn.out = adapter_cls(xf.attn.out, **kwargs)

    if target in ("all", "ffn"):
      if isinstance(xf.ff0, nn.Linear):
        xf.ff0 = adapter_cls(xf.ff0, **kwargs)
      if isinstance(xf.ff1, nn.Linear):
        xf.ff1 = adapter_cls(xf.ff1, **kwargs)

  # 4. Optionally unfreeze output heads.
  if config.train_output_head:
    for p in model.output_projection_point.parameters():
      p.requires_grad = True
    for p in model.output_projection_quantiles.parameters():
      p.requires_grad = True

  return model


def merge_adapters(model: nn.Module) -> nn.Module:
  """Fold all adapter weights back into base ``nn.Linear`` layers.

  After merging, the model has standard ``nn.Linear`` modules and can be
  used for normal inference or saved as a regular checkpoint.
  """
  for layer_idx in range(model.x):
    xf = model.stacked_xf[layer_idx]

    for attr in ("qkv_proj", "out"):
      layer = getattr(xf.attn, attr, None)
      if isinstance(layer, (LoRALinear, DoRALinear)):
        setattr(xf.attn, attr, layer.merge_weights())
    for attr in ("query", "key", "value"):
      layer = getattr(xf.attn, attr, None)
      if isinstance(layer, (LoRALinear, DoRALinear)):
        setattr(xf.attn, attr, layer.merge_weights())
    for attr in ("ff0", "ff1"):
      layer = getattr(xf, attr, None)
      if isinstance(layer, (LoRALinear, DoRALinear)):
        setattr(xf, attr, layer.merge_weights())

  # Unfreeze everything so the merged model can be retrained if desired.
  for p in model.parameters():
    p.requires_grad = True

  return model


# ---------------------------------------------------------------------------
# Save / load adapter-only weights
# ---------------------------------------------------------------------------


def get_adapter_params(model: nn.Module) -> Dict[str, torch.Tensor]:
  """Return an ``OrderedDict`` of all trainable (adapter) parameters."""
  return OrderedDict(
    (n, p.data) for n, p in model.named_parameters() if p.requires_grad
  )


def save_adapter_weights(model: nn.Module, path: str) -> None:
  """Save adapter weights to a ``safetensors`` file."""
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  save_file(get_adapter_params(model), path)


def load_adapter_weights(model: nn.Module, path: str) -> None:
  """Load adapter weights from a ``safetensors`` file.

  The model must already have adapters injected (via ``inject_adapters``)
  before calling this function.
  """
  tensors = load_file(path, device="cpu")
  trainable = {n for n, p in model.named_parameters() if p.requires_grad}
  missing = trainable - set(tensors.keys())
  if missing:
    raise ValueError(f"Adapter checkpoint is missing keys: {missing}")

  state = model.state_dict()
  state.update(tensors)
  model.load_state_dict(state, strict=True)
