#!/usr/bin/env python3
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

"""CLI entry-point for TimesFM 2.5 PEFT fine-tuning.

Single-GPU::

    python peft/finetune.py --data_path data.csv --value_col y

Multi-GPU (4 GPUs)::

    torchrun --nproc_per_node=4 peft/finetune.py --data_path data.csv --value_col y
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
  datefmt="%H:%M:%S",
)
logger = logging.getLogger("peft.finetune")


def parse_args(argv=None):
  p = argparse.ArgumentParser(
    description="Fine-tune TimesFM 2.5 with LoRA / DoRA (multi-GPU ready)."
  )

  # -- Model ---------------------------------------------------------------
  g = p.add_argument_group("Model")
  g.add_argument(
    "--model_id",
    default="google/timesfm-2.5-200m-pytorch",
    help="HuggingFace repo-id or local directory for the base model.",
  )

  # -- Data ----------------------------------------------------------------
  g = p.add_argument_group("Data")
  g.add_argument("--data_path", required=True, help="Path to a CSV file.")
  g.add_argument(
    "--id_col",
    default=None,
    help="Column identifying individual time series (long format).",
  )
  g.add_argument(
    "--value_col",
    default=None,
    help="Column with the values to forecast (long format).",
  )
  g.add_argument("--context_len", type=int, default=512)
  g.add_argument(
    "--horizon_len",
    type=int,
    default=128,
    help="Prediction horizon (max 128 for single-step training).",
  )
  g.add_argument(
    "--stride",
    type=int,
    default=32,
    help="Stride for the sliding-window dataset.",
  )
  g.add_argument(
    "--val_split",
    type=float,
    default=0.2,
    help="Fraction of each series reserved for validation.",
  )

  # -- Adapter -------------------------------------------------------------
  g = p.add_argument_group("Adapter")
  g.add_argument(
    "--adapter_type",
    choices=["lora", "dora"],
    default="lora",
  )
  g.add_argument("--lora_rank", type=int, default=8)
  g.add_argument("--lora_alpha", type=float, default=16.0)
  g.add_argument("--lora_dropout", type=float, default=0.0)
  g.add_argument(
    "--target_modules",
    choices=["all", "attention", "ffn"],
    default="all",
  )
  g.add_argument(
    "--num_adapter_layers",
    type=int,
    default=0,
    help="Only adapt the last N transformer layers (0 = all 20). "
    "Advisor recommends 2-4 for financial data.",
  )
  g.add_argument(
    "--train_output_head",
    action="store_true",
    help="Also train the output projection heads.",
  )

  # -- Training ------------------------------------------------------------
  g = p.add_argument_group("Training")
  g.add_argument("--num_epochs", type=int, default=10)
  g.add_argument("--batch_size", type=int, default=32)
  g.add_argument("--learning_rate", type=float, default=1e-4)
  g.add_argument("--weight_decay", type=float, default=0.01)
  g.add_argument("--gradient_clip_norm", type=float, default=1.0)
  g.add_argument("--warmup_ratio", type=float, default=0.05)
  g.add_argument(
    "--mixed_precision",
    choices=["no", "fp16", "bf16"],
    default="no",
  )
  g.add_argument("--gradient_checkpointing", action="store_true")
  g.add_argument("--use_quantile_loss", action="store_true")
  g.add_argument("--quantile_loss_weight", type=float, default=0.5)

  # -- Logging / checkpointing --------------------------------------------
  g = p.add_argument_group("Logging")
  g.add_argument("--use_wandb", action="store_true")
  g.add_argument("--wandb_project", default="timesfm-2.5-peft")
  g.add_argument("--log_every_n_steps", type=int, default=50)
  g.add_argument("--checkpoint_dir", default="./peft_checkpoints")
  g.add_argument("--save_every_n_epochs", type=int, default=1)
  g.add_argument("--early_stopping_patience", type=int, default=5)

  # -- Misc ----------------------------------------------------------------
  g = p.add_argument_group("Misc")
  g.add_argument("--num_workers", type=int, default=4)
  g.add_argument("--seed", type=int, default=42)

  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)

  # Lazy imports so --help is fast.
  from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

  from .config import PEFTConfig
  from .data import TimeSeriesDataset
  from .trainer import PEFTTrainer

  # -- Load model ----------------------------------------------------------
  logger.info("Loading base model from %s …", args.model_id)
  wrapper = TimesFM_2p5_200M_torch.from_pretrained(
    args.model_id, torch_compile=False
  )

  # -- Build config --------------------------------------------------------
  config = PEFTConfig(
    adapter_type=args.adapter_type,
    lora_rank=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=args.target_modules,
    num_adapter_layers=args.num_adapter_layers,
    train_output_head=args.train_output_head,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    gradient_clip_norm=args.gradient_clip_norm,
    warmup_ratio=args.warmup_ratio,
    context_len=args.context_len,
    horizon_len=args.horizon_len,
    use_quantile_loss=args.use_quantile_loss,
    quantile_loss_weight=args.quantile_loss_weight,
    mixed_precision=args.mixed_precision,
    gradient_checkpointing=args.gradient_checkpointing,
    use_wandb=args.use_wandb,
    wandb_project=args.wandb_project,
    log_every_n_steps=args.log_every_n_steps,
    checkpoint_dir=args.checkpoint_dir,
    save_every_n_epochs=args.save_every_n_epochs,
    early_stopping_patience=args.early_stopping_patience,
    num_workers=args.num_workers,
    seed=args.seed,
  )

  # -- Load data -----------------------------------------------------------
  logger.info("Reading data from %s …", args.data_path)
  df = pd.read_csv(args.data_path)

  # Parse series from DataFrame.
  if args.id_col and args.value_col:
    all_series = [
      grp[args.value_col].to_numpy(dtype=np.float32)
      for _, grp in df.groupby(args.id_col, sort=False)
    ]
  elif args.value_col:
    all_series = [df[args.value_col].to_numpy(dtype=np.float32)]
  else:
    all_series = [
      df[c].to_numpy(dtype=np.float32)
      for c in df.select_dtypes(include="number").columns
    ]

  # Train / val split (tail of each series → val).
  train_series, val_series = [], []
  for s in all_series:
    split_idx = max(1, int(len(s) * (1 - args.val_split)))
    train_series.append(s[:split_idx])
    val_series.append(s[split_idx - config.context_len :])  # overlap for context

  train_ds = TimeSeriesDataset(
    train_series,
    context_len=config.context_len,
    horizon_len=config.horizon_len,
    stride=args.stride,
  )
  val_ds = TimeSeriesDataset(
    val_series,
    context_len=config.context_len,
    horizon_len=config.horizon_len,
    stride=config.horizon_len,  # non-overlapping for val
  )

  logger.info(
    "Dataset: %d train windows, %d val windows", len(train_ds), len(val_ds)
  )

  # -- Train ---------------------------------------------------------------
  trainer = PEFTTrainer(wrapper.model, config)
  history = trainer.fit(train_ds, val_ds)

  # -- Save final adapter --------------------------------------------------
  final_path = f"{config.checkpoint_dir}/final_adapter.safetensors"
  trainer.save_adapter(final_path)
  logger.info("Final adapter saved → %s", final_path)

  return history


if __name__ == "__main__":
  main()
