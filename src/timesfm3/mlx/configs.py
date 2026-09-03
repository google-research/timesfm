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

"""Configuration for the MLX TimesFM3 model.

Mirrors the fields the PyTorch backend reads from ``config.json`` (see ``torch/configs.py``),
flattened to what the MLX model needs. ``from_hf_config`` maps a checkpoint's ``config.json`` onto
this dataclass.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TimesFM3MlxConfig:
  """Architecture hyper-parameters for the MLX TimesFM3 model."""

  input_patch_len: int = 32
  output_patch_len: int = 64
  model_dims: int = 1280
  hidden_dims: int = 1280
  num_layers: int = 20
  num_heads: int = 16
  quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  use_variate_attention: bool = True
  use_stitching: bool = True
  use_linear_detrending: bool = True
  linear_detrending_threshold: float = 0.5
  value_clip: float = 1e20

  @property
  def head_dim(self) -> int:
    return self.model_dims // self.num_heads

  @property
  def num_quantiles(self) -> int:
    return len(self.quantiles)

  @property
  def rolls(self) -> int:
    return self.output_patch_len // self.input_patch_len

  @classmethod
  def from_hf_config(cls, cfg: dict) -> "TimesFM3MlxConfig":
    """Build a config from a checkpoint's ``config.json`` dictionary."""
    if cfg.get("use_frozen_running_stats", False):
      # Torch freezes the running RevIN stats at the context boundary; the MLX
      # backend does not implement that yet, so a checkpoint that needs it would
      # diverge on any unmasked horizon input (past-future covariates). The
      # public 3.0 checkpoint sets this to False. Fail loudly instead.
      raise NotImplementedError(
        "The MLX backend does not implement use_frozen_running_stats=True; "
        "use the torch backend for this checkpoint."
      )
    transformer = cfg.get("transformer_config", {})
    inner = transformer.get("transformer", {})
    return cls(
      input_patch_len=cfg.get("input_patch_len", 32),
      output_patch_len=cfg.get("output_patch_len", 64),
      model_dims=inner.get("model_dims", 1280),
      hidden_dims=inner.get("hidden_dims", 1280),
      num_layers=transformer.get("num_layers", 20),
      num_heads=inner.get("num_heads", 16),
      quantiles=tuple(cfg.get("quantiles", cls.quantiles)),
      use_variate_attention=cfg.get("use_variate_attention", True),
      use_stitching=cfg.get("use_stitching", True),
      use_linear_detrending=cfg.get("use_linear_detrending", True),
      linear_detrending_threshold=cfg.get("linear_detrending_threshold", 0.5),
      value_clip=cfg.get("value_clip", 1e20),
    )
