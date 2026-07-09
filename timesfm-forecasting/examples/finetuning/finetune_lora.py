#!/usr/bin/env python3
"""Fine-tune TimesFM 2.5 with LoRA using HuggingFace Transformers + PEFT.

This script demonstrates parameter-efficient fine-tuning of TimesFM 2.5 on a
retail demand forecasting dataset (weekly store sales).  It uses the HuggingFace
Transformers checkpoint and the standard PEFT library for LoRA adapters.

The approach is based on the fine-tuning workflow by @kashif at HuggingFace:
https://github.com/huggingface/notebooks/blob/main/examples/timesfm2_5.ipynb

The dataset is the same one used in the Chronos-2 quickstart notebook.  Each
store has ~120 weekly data points.  The goal is to forecast the next 13 weeks
(one quarter) of sales per store.

Requirements:
    pip install transformers accelerate peft pandas pyarrow scikit-learn

Usage:
    python finetune_lora.py [OPTIONS]

    Options:
        --model_id       HuggingFace model ID (default: google/timesfm-2.5-200m-transformers)
        --context_len    Context length for training windows (default: 64, must be multiple of 32)
        --horizon_len    Forecast horizon in time steps (default: 13)
        --epochs         Number of training epochs (default: 10)
        --batch_size     Training batch size (default: 32)
        --lr             Learning rate (default: 1e-4)
        --lora_r         LoRA rank (default: 4)
        --lora_alpha     LoRA alpha (default: 8)
        --lora_dropout   LoRA dropout (default: 0.05)
        --num_samples    Number of random training windows to pre-sample (default: 5000)
        --output_dir     Directory to save the LoRA adapter (default: timesfm2_5-retail-lora)
        --seed           Random seed (default: 42)
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesRandomWindowDataset(Dataset):
    """Random-window dataset for time series fine-tuning.

    Pre-samples random (series, split-point) windows similar to Chronos-2's
    random slicing.  Each window has a full *context_len* context (no
    zero-padding) to avoid corrupting TimesFM's internal RevIN normalisation
    statistics.

    No external normalisation is needed — TimesFM handles instance
    normalisation internally.  The loss is computed in the original data scale.
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
        num_samples: int = 5000,
        seed: int = 42,
    ):
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.samples: list[tuple[int, int]] = []

        rng = np.random.default_rng(seed)
        min_len = context_len + horizon_len
        valid = [i for i, s in enumerate(series_list) if len(s) >= min_len]
        if not valid:
            raise ValueError(
                f"No series long enough for context_len={context_len} + "
                f"horizon_len={horizon_len}. Shortest series: "
                f"{min(len(s) for s in series_list)}"
            )

        for _ in range(num_samples):
            idx = rng.choice(valid)
            series = series_list[idx]
            max_start = len(series) - min_len
            start = rng.integers(0, max_start + 1)
            self.samples.append((idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        idx, start = self.samples[i]
        series = self.series_list[idx]
        end = start + self.context_len + self.horizon_len

        context = torch.tensor(
            series[start : start + self.context_len], dtype=torch.float32
        )
        target = torch.tensor(
            series[start + self.context_len : end], dtype=torch.float32
        )
        return context, target


class TimeSeriesLastWindowDataset(Dataset):
    """Validation dataset using the last window of each series."""

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
    ):
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        min_len = context_len + horizon_len
        for s in series_list:
            if len(s) >= min_len:
                ctx = torch.tensor(s[-min_len:-horizon_len], dtype=torch.float32)
                tgt = torch.tensor(s[-horizon_len:], dtype=torch.float32)
                self.items.append((ctx, tgt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_retail_sales(
    context_len: int,
    horizon_len: int,
    num_samples: int,
    seed: int,
) -> tuple[TimeSeriesRandomWindowDataset, TimeSeriesLastWindowDataset]:
    """Download and prepare the retail sales dataset.

    This is the same dataset used in the Chronos-2 quickstart notebook and
    in @kashif's TimesFM 2.5 fine-tuning example.  Each store has ~120 weekly
    data points; the target column is ``Sales``.

    Returns train dataset and val dataset.
    """
    logger.info("Loading retail sales dataset …")
    sales_train_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/"
        "retail_sales/train.parquet"
    )
    target = "Sales"

    all_series: list[np.ndarray] = []
    for _, group in sales_train_df.groupby("id"):
        values = group[target].values.astype(np.float32)
        if len(values) >= context_len + horizon_len:
            all_series.append(values)

    logger.info(
        "Valid stores: %d (need >= %d data points)",
        len(all_series),
        context_len + horizon_len,
    )

    train_ds = TimeSeriesRandomWindowDataset(
        all_series, context_len, horizon_len, num_samples=num_samples, seed=seed
    )
    val_ds = TimeSeriesLastWindowDataset(all_series, context_len, horizon_len)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", args.model_id)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    horizon_len = args.horizon_len
    context_len = min(args.context_len, model.config.context_length)

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    train_ds, val_ds = load_retail_sales(
        context_len, horizon_len, num_samples=args.num_samples, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    logger.info(
        "Train samples: %d (%d batches) | Val samples: %d",
        len(train_ds),
        len(train_loader),
        len(val_ds),
    )

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for context, target_vals in train_loader:
            context = context.to(device)
            target_vals = target_vals.to(device)

            outputs = model(
                past_values=context,
                future_values=target_vals,
                forecast_context_len=context_len,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for context, target_vals in val_loader:
                context = context.to(device)
                target_vals = target_vals.to(device)
                outputs = model(
                    past_values=context,
                    future_values=target_vals,
                    forecast_context_len=context_len,
                )
                val_loss += outputs.loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        logger.info(
            "Epoch %d/%d (%d steps) — train loss: %.4f, val loss: %.4f",
            epoch,
            args.epochs,
            n_batches,
            avg_train_loss,
            avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(args.output_dir)
            logger.info("  ✓ saved best adapter → %s", args.output_dir)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    """Compare zero-shot vs fine-tuned on a subset of stores."""
    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading base model …")
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()
    horizon_len = args.horizon_len
    context_len = min(args.context_len, base_model.config.context_length)

    logger.info("Loading LoRA adapter from %s …", args.output_dir)
    ft_model = PeftModel.from_pretrained(base_model, args.output_dir)
    ft_model.eval()

    # --- Load data ---
    sales_train_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/"
        "retail_sales/train.parquet"
    )
    sales_test_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/"
        "retail_sales/test.parquet"
    )
    target = "Sales"

    store_ids = sales_train_df["id"].unique()[:8]

    base_maes: list[float] = []
    ft_maes: list[float] = []

    for store_id in store_ids:
        store_train = (
            sales_train_df[sales_train_df["id"] == store_id][target]
            .values.astype(np.float32)
        )
        store_test = (
            sales_test_df[sales_test_df["id"] == store_id][target]
            .values.astype(np.float32)
        )
        ground_truth = store_test[:horizon_len]
        if len(ground_truth) < horizon_len or len(store_train) < context_len:
            continue

        test_input = torch.tensor(
            store_train[-context_len:], dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            base_out = base_model(past_values=test_input)
            ft_out = ft_model(past_values=test_input)

        base_forecast = base_out.mean_predictions[0, :horizon_len].float().cpu().numpy()
        ft_forecast = ft_out.mean_predictions[0, :horizon_len].float().cpu().numpy()

        base_mae = float(np.abs(base_forecast - ground_truth).mean())
        ft_mae = float(np.abs(ft_forecast - ground_truth).mean())
        base_maes.append(base_mae)
        ft_maes.append(ft_mae)

        logger.info(
            "Store %s — zero-shot MAE: %.2f, LoRA MAE: %.2f",
            store_id,
            base_mae,
            ft_mae,
        )

    if base_maes:
        avg_base = np.mean(base_maes)
        avg_ft = np.mean(ft_maes)
        improvement = (avg_base - avg_ft) / avg_base * 100
        logger.info("Average zero-shot MAE: %.2f", avg_base)
        logger.info("Average LoRA MAE:      %.2f", avg_ft)
        logger.info("Improvement:           %.1f%%", improvement)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune TimesFM 2.5 with LoRA (Transformers + PEFT)"
    )
    p.add_argument(
        "--model_id",
        default="google/timesfm-2.5-200m-transformers",
        help="HuggingFace model ID",
    )
    p.add_argument("--context_len", type=int, default=64)
    p.add_argument("--horizon_len", type=int, default=13)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--output_dir", default="timesfm2_5-retail-lora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only run evaluation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.eval_only:
        train(args)

    if os.path.isdir(args.output_dir):
        evaluate(args)
    else:
        logger.warning(
            "No adapter found at %s — skipping evaluation.", args.output_dir
        )


if __name__ == "__main__":
    main()
