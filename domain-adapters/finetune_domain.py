#!/usr/bin/env python3
"""Fine-tune a domain-specialised LoRA adapter for TimesFM 2.5.

The base model is never modified. Each domain (crypto, stocks, ...) gets its
own small adapter in adapters/<domain>/ that is loaded on top of the base model.

Data layout (produced by fetch_crypto.py, or any fetcher with the same layout):
    <data_dir>/<interval>/<SERIES>.parquet   with columns: timestamp, <column>

Time-based splits across ALL series share the same calendar cutoffs, so no
future information leaks between correlated assets:
    train:  targets strictly before --val_start
    val:    forecast origins in [--val_start, --test_start)   (picks best checkpoint)
    test:   forecast origins at or after --test_start          (reported, never trained on)

Usage:
    python finetune_domain.py --domain crypto
    python finetune_domain.py --domain crypto --steps 8000 --lora_r 16 --lora_alpha 32
    python finetune_domain.py --domain crypto --eval_only
"""

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

BASE_MODEL_ID = "google/timesfm-2.5-200m-transformers"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Series:
  name: str
  interval: str
  times: np.ndarray  # datetime64[ns], UTC
  values: np.ndarray  # float32


def load_series(data_dir: Path, intervals: list[str], column: str, min_len: int) -> list[Series]:
  out: list[Series] = []
  for interval in intervals:
    files = sorted((data_dir / interval).glob("*.parquet"))
    if not files:
      raise FileNotFoundError(f"No parquet files in {data_dir / interval}")
    for f in files:
      df = pd.read_parquet(f, columns=["timestamp", column]).dropna()
      df = df[df[column] > 0]
      if len(df) < min_len:
        continue
      times = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
      out.append(Series(f.stem, interval, times, df[column].to_numpy(np.float32)))
  return out


def origins_in_range(s: Series, lo, hi, ctx: int, hor: int) -> np.ndarray:
  """Indices i such that context = values[i-ctx:i], target = values[i:i+hor],
  and the first target timestamp falls within [lo, hi)."""
  idx = np.arange(ctx, len(s.values) - hor + 1)
  t = s.times[idx]
  mask = np.ones_like(idx, dtype=bool)
  if lo is not None:
    mask &= t >= lo
  if hi is not None:
    mask &= t < hi
  return idx[mask]


class RandomWindowSampler:
  """Draws fresh random training windows every batch.

  Sampling is balanced: pick an interval uniformly, then a series uniformly,
  then a random position. Otherwise hourly data (24x more points than daily)
  would drown out everything else.
  """

  def __init__(self, series: list[Series], val_start, ctx: int, hor: int, seed: int):
    self.ctx, self.hor = ctx, hor
    self.rng = np.random.default_rng(seed)
    self.by_interval: dict[str, list[tuple[np.ndarray, int]]] = {}
    for s in series:
      # Train targets must END before val_start.
      n_train = int(np.searchsorted(s.times, val_start))
      if n_train >= ctx + hor:
        self.by_interval.setdefault(s.interval, []).append((s.values[:n_train], n_train))
    if not self.by_interval:
      raise ValueError("No series has enough pre-val_start history for one training window.")
    self.intervals = sorted(self.by_interval)

  def summary(self) -> str:
    return ", ".join(
        f"{k}: {len(v)} series / {sum(n for _, n in v):,} pts" for k, v in self.by_interval.items()
    )

  def batch(self, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
    ctxs, tgts = [], []
    for _ in range(bs):
      pool = self.by_interval[self.intervals[self.rng.integers(len(self.intervals))]]
      vals, n = pool[self.rng.integers(len(pool))]
      i = self.rng.integers(self.ctx, n - self.hor + 1)
      ctxs.append(vals[i - self.ctx : i])
      tgts.append(vals[i : i + self.hor])
    return torch.from_numpy(np.stack(ctxs)), torch.from_numpy(np.stack(tgts))


def fixed_windows(series, lo, hi, ctx, hor, max_per_series, stride):
  """Deterministic evaluation windows, grouped by interval."""
  groups: dict[str, tuple[list, list]] = {}
  for s in series:
    idx = origins_in_range(s, lo, hi, ctx, hor)[::stride]
    if len(idx) > max_per_series:
      idx = idx[np.linspace(0, len(idx) - 1, max_per_series).round().astype(int)]
    c, t = groups.setdefault(s.interval, ([], []))
    for i in idx:
      c.append(s.values[i - ctx : i])
      t.append(s.values[i : i + hor])
  return {k: (np.stack(c), np.stack(t)) for k, (c, t) in groups.items() if c}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, contexts: np.ndarray, hor: int, bs: int, device) -> np.ndarray:
  """Returns full predictions [N, hor, 10]: index 0 = mean, 1..9 = q0.1..q0.9."""
  outs = []
  for i in range(0, len(contexts), bs):
    x = torch.from_numpy(contexts[i : i + bs]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
      o = model(past_values=x, forecast_context_len=x.shape[1])
    outs.append(o.full_predictions[:, :hor].float().cpu().numpy())
  return np.concatenate(outs)


def metrics(full: np.ndarray, ctx: np.ndarray, tgt: np.ndarray) -> dict:
  """Scale-free metrics, every window normalised by its last observed value."""
  last = ctx[:, -1:]
  y = tgt / last
  pred = full[..., 0] / last
  q = full[..., 1:] / last[..., None]
  naive = np.ones_like(y)  # "price stays where it is"

  mae = np.abs(pred - y).mean()
  naive_mae = np.abs(naive - y).mean()
  # Pinball loss averaged over the 9 quantiles (lower = better calibrated distribution).
  levels = np.linspace(0.1, 0.9, 9)
  err = y[..., None] - q
  pinball = np.maximum(levels * err, (levels - 1) * err).mean()
  # Direction at the end of the horizon (excluding exact ties).
  moved = np.sign(y[:, -1] - 1)
  keep = moved != 0
  direction = (np.sign(pred[keep, -1] - 1) == moved[keep]).mean()
  coverage80 = ((y >= q[..., 0]) & (y <= q[..., 8])).mean()
  return {
      "windows": int(len(y)),
      "MAE_%": float(100 * mae),
      "naive_MAE_%": float(100 * naive_mae),
      "skill_vs_naive_%": float(100 * (1 - mae / naive_mae)),
      "pinball_%": float(100 * pinball),
      "direction_acc_%": float(100 * direction),
      "coverage80_%": float(100 * coverage80),
  }


def evaluate(model, windows: dict, hor, bs, device, label: str, adapter_on: bool) -> dict:
  results = {}
  for interval, (c, t) in sorted(windows.items()):
    if adapter_on:
      full = predict(model, c, hor, bs, device)
    else:
      with model.disable_adapter():
        full = predict(model, c, hor, bs, device)
    results[interval] = metrics(full, c, t)
  return results


def print_comparison(base: dict, tuned: dict, title: str) -> None:
  keys = ["MAE_%", "skill_vs_naive_%", "pinball_%", "direction_acc_%", "coverage80_%"]
  better = {"MAE_%": -1, "skill_vs_naive_%": 1, "pinball_%": -1, "direction_acc_%": 1}
  print(f"\n=== {title} ===")
  for interval in sorted(base):
    b, t = base[interval], tuned[interval]
    print(f"[{interval}]  {b['windows']} windows  |  naive MAE {b['naive_MAE_%']:.3f}%")
    print(f"  {'metric':<18}{'base':>10}{'crypto-LoRA':>13}")
    for k in keys:
      mark = ""
      if k in better:
        mark = "  better" if (t[k] - b[k]) * better[k] > 0 else ""
      elif k == "coverage80_%":
        mark = "  better" if abs(t[k] - 80) < abs(b[k] - 80) else ""
      print(f"  {k:<18}{b[k]:>10.3f}{t[k]:>13.3f}{mark}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--domain", required=True, help="Name of the adapter, e.g. crypto or stocks")
  p.add_argument("--data_dir", default=None, help="Defaults to data/<domain>")
  p.add_argument("--intervals", nargs="+", default=["1h", "1d"])
  p.add_argument("--column", default="close")
  p.add_argument("--val_start", default="2025-09-01")
  p.add_argument("--test_start", default="2026-03-01")
  p.add_argument("--context_len", type=int, default=512, help="Multiple of 32")
  p.add_argument("--horizon_len", type=int, default=24, help="<= 128")
  p.add_argument("--steps", type=int, default=800)
  p.add_argument("--batch_size", type=int, default=256, help="Step time is overhead-bound, so big batches are ~free")
  p.add_argument("--lr", type=float, default=5e-5)
  p.add_argument("--loss", choices=["relative", "native"], default="relative",
                 help="relative = error as %% of last price (recommended for prices)")
  p.add_argument("--warmup", type=int, default=200)
  p.add_argument("--lora_r", type=int, default=8)
  p.add_argument("--lora_alpha", type=int, default=16)
  p.add_argument("--lora_dropout", type=float, default=0.05)
  p.add_argument("--eval_every", type=int, default=200)
  p.add_argument("--val_windows_per_series", type=int, default=48)
  p.add_argument("--test_windows_per_series", type=int, default=200)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--eval_only", action="store_true")
  args = p.parse_args()

  assert args.context_len % 32 == 0, "context_len must be a multiple of 32"
  assert args.horizon_len <= 128, "TimesFM 2.5 output head covers 128 steps"

  from peft import LoraConfig, PeftModel, get_peft_model
  from transformers import TimesFm2_5ModelForPrediction

  torch.manual_seed(args.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data_dir = Path(args.data_dir or f"data/{args.domain}")
  out_dir = Path("adapters") / args.domain
  val_start = np.datetime64(pd.Timestamp(args.val_start))
  test_start = np.datetime64(pd.Timestamp(args.test_start))
  ctx, hor = args.context_len, args.horizon_len

  # ---- data ----
  series = load_series(data_dir, args.intervals, args.column, min_len=ctx + hor)
  log.info("Loaded %d series from %s", len(series), data_dir)
  sampler = RandomWindowSampler(series, val_start, ctx, hor, args.seed)
  log.info("Train pool (before %s): %s", args.val_start, sampler.summary())
  val_w = fixed_windows(series, val_start, test_start, ctx, hor, args.val_windows_per_series, stride=hor)
  test_w = fixed_windows(series, test_start, None, ctx, hor, args.test_windows_per_series, stride=hor)
  log.info("Val windows: %s", {k: len(v[0]) for k, v in val_w.items()})
  log.info("Test windows (from %s): %s", args.test_start, {k: len(v[0]) for k, v in test_w.items()})

  # ---- model ----
  # fp32 master weights + bf16 autocast: trains stably, and 200M params is tiny for 16 GB.
  base = TimesFm2_5ModelForPrediction.from_pretrained(BASE_MODEL_ID, dtype=torch.float32).to(device)

  if args.eval_only:
    model = PeftModel.from_pretrained(base, out_dir)
  else:
    model = get_peft_model(
        base,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    train(model, sampler, val_w, args, device, out_dir)
    # Reload the best checkpoint for the final test.
    model = PeftModel.from_pretrained(
        TimesFm2_5ModelForPrediction.from_pretrained(BASE_MODEL_ID, dtype=torch.float32).to(device), out_dir
    )

  model.eval()
  base_res = evaluate(model, test_w, hor, 256, device, "base", adapter_on=False)
  tuned_res = evaluate(model, test_w, hor, 256, device, "tuned", adapter_on=True)
  print_comparison(base_res, tuned_res, f"TEST (unseen period from {args.test_start}), horizon {hor} steps")
  (out_dir / "test_results.json").write_text(json.dumps({"base": base_res, "tuned": tuned_res}, indent=2))


def relative_loss(model, x: torch.Tensor, y: torch.Tensor, loss_type: str) -> torch.Tensor:
  """Training objective.

  'relative' (default): errors measured as a fraction of the last observed price,
  i.e. exactly what a trader cares about. Pinball loss over the 9 quantiles plus
  L1 on the point forecast.

  'native': the model's built-in loss, normalised by the context window's std.
  Warning: for assets that trended strongly within the context, a large %-miss
  looks small in this space, which teaches the adapter to over-extrapolate.
  """
  out = model(past_values=x, future_values=y if loss_type == "native" else None, forecast_context_len=x.shape[1])
  if loss_type == "native":
    return out.loss
  last = x[:, -1:].float()
  pred = out.full_predictions[:, : y.shape[1]].float() / last[..., None]
  target = (y.float() / last)[..., None]
  levels = torch.linspace(0.1, 0.9, 9, device=x.device)
  err = target - pred[..., 1:]
  pinball = torch.maximum(levels * err, (levels - 1) * err).mean()
  l1 = (target[..., 0] - pred[..., 0]).abs().mean()
  return 100 * (pinball + l1)  # in % units, so logged numbers are readable


def val_loss(model, windows, bs, device, loss_type) -> float:
  """Mean over intervals of the mean loss within each interval, so 1h windows
  (many) don't drown out 1d windows (few)."""
  model.eval()
  per_interval = []
  with torch.no_grad():
    for c, t in windows.values():
      total = 0.0
      for i in range(0, len(c), bs):
        x = torch.from_numpy(c[i : i + bs]).to(device)
        y = torch.from_numpy(t[i : i + bs]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
          total += relative_loss(model, x, y, loss_type).item() * len(x)
      per_interval.append(total / len(c))
  model.train()
  return float(np.mean(per_interval))


def train(model, sampler, val_w, args, device, out_dir: Path) -> None:
  params = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
  sched = torch.optim.lr_scheduler.LambdaLR(
      opt,
      lambda s: min(1.0, (s + 1) / args.warmup)
      * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))),
  )

  best = val_loss(model, val_w, 256, device, args.loss)
  with model.disable_adapter():
    base_val = val_loss(model, val_w, 256, device, args.loss)
  log.info("Val loss before training: %.4f (base model: %.4f)", best, base_val)
  out_dir.mkdir(parents=True, exist_ok=True)
  model.save_pretrained(out_dir)

  model.train()
  running, t0 = 0.0, time.time()
  for step in range(1, args.steps + 1):
    x, y = sampler.batch(args.batch_size)
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
      loss = relative_loss(model, x, y, args.loss)
    if not torch.isfinite(loss):
      log.warning("Non-finite loss at step %d - skipping batch", step)
      opt.zero_grad(set_to_none=True)
      continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    running += loss.item()

    if step % args.eval_every == 0 or step == args.steps:
      v = val_loss(model, val_w, 256, device, args.loss)
      tag = ""
      if v < best:
        best = v
        model.save_pretrained(out_dir)
        tag = "  <- saved"
      log.info(
          "step %5d/%d  train %.4f  val %.4f (base %.4f)  lr %.1e  %.1f it/s%s",
          step, args.steps, running / args.eval_every, v, base_val,
          sched.get_last_lr()[0], args.eval_every / (time.time() - t0), tag,
      )
      running, t0 = 0.0, time.time()

  meta = {k: v for k, v in vars(args).items() if k != "eval_only"}
  meta.update(base_model=BASE_MODEL_ID, best_val_loss=best, base_val_loss=base_val)
  (out_dir / "training_config.json").write_text(json.dumps(meta, indent=2))
  log.info("Done. Best val loss %.4f (base %.4f). Adapter: %s", best, base_val, out_dir)


if __name__ == "__main__":
  main()
