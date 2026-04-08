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

"""Configuration for the TimesFM 2.5 PEFT fine-tuning pipeline."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class PEFTConfig:
  """Full configuration for PEFT fine-tuning of TimesFM 2.5.

  Attributes:
    adapter_type: Type of adapter — "lora" or "dora".
    lora_rank: Rank of the low-rank decomposition.
    lora_alpha: Scaling factor (effective lr multiplier = alpha / rank).
    lora_dropout: Dropout applied to the LoRA path.
    target_modules: Which layers to adapt — "all", "attention", or "ffn".
    num_adapter_layers: How many transformer layers (from the top) to adapt.
      0 means all 20 layers.  E.g. 4 means only layers 16-19 get adapters.
      The advisor recommends 2–4 for financial data to avoid overfitting.
    train_output_head: Whether to also unfreeze and train the output
      projection heads (point + quantile).

    learning_rate: Peak learning rate for AdamW.
    weight_decay: L2 regularization coefficient.
    num_epochs: Number of training epochs.
    batch_size: Per-device batch size.
    gradient_clip_norm: Max gradient norm for clipping.
    warmup_ratio: Fraction of total steps used for linear warmup.

    context_len: Context window length (padded up to a multiple of 32).
    horizon_len: Prediction horizon (must be <= 128 for single-step training).

    use_quantile_loss: Whether to add pinball loss on quantile channels.
    quantile_loss_weight: Relative weight of the quantile loss term.

    mixed_precision: AMP dtype — "no", "fp16", or "bf16".
    gradient_checkpointing: Trade compute for memory in the transformer stack.

    use_wandb: Enable Weights & Biases logging (rank-0 only).
    wandb_project: W&B project name.
    log_every_n_steps: Console / W&B logging frequency.

    checkpoint_dir: Directory for adapter checkpoints.
    save_every_n_epochs: Checkpoint save frequency.
    early_stopping_patience: Epochs without val-loss improvement before stop.

    num_workers: DataLoader workers per process.
    seed: Random seed for reproducibility.
  """

  # --- Adapter ---
  adapter_type: Literal["lora", "dora"] = "lora"
  lora_rank: int = 8
  lora_alpha: float = 16.0
  lora_dropout: float = 0.0
  target_modules: Literal["all", "attention", "ffn"] = "all"
  num_adapter_layers: int = 0  # 0 = all 20 layers; N > 0 = only last N layers
  train_output_head: bool = False

  # --- Optimiser ---
  learning_rate: float = 1e-4
  weight_decay: float = 0.01
  num_epochs: int = 10
  batch_size: int = 32
  gradient_clip_norm: float = 1.0
  warmup_ratio: float = 0.05

  # --- Data ---
  context_len: int = 512
  horizon_len: int = 128

  # --- Loss ---
  use_quantile_loss: bool = False
  quantile_loss_weight: float = 0.5

  # --- Performance ---
  mixed_precision: Literal["no", "fp16", "bf16"] = "no"
  gradient_checkpointing: bool = False

  # --- Logging ---
  use_wandb: bool = False
  wandb_project: str = "timesfm-2.5-peft"
  log_every_n_steps: int = 50

  # --- Checkpointing ---
  checkpoint_dir: str = "./peft_checkpoints"
  save_every_n_epochs: int = 1
  early_stopping_patience: int = 5

  # --- Misc ---
  num_workers: int = 4
  seed: int = 42
