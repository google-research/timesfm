#!/usr/bin/env python3
"""Fine-tune a candle-aware Horizon model (reads close + high/low/volume).

Starts from an existing adapter (default: horizon-c2) and adds the candle
encoder from ohlcv.py. The encoder starts at zero, so step 0 == the init model.

    python finetune_ohlcv.py --name horizon-c3                    # with high/low/volume
    python finetune_ohlcv.py --name c2-control --no_candle        # same training, close only
    python finetune_ohlcv.py --name horizon-c3 --eval_only

The --no_candle control matters: without it you can't tell whether a gain came
from high/low/volume or just from extra training steps.
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune_domain import metrics
from ohlcv import CandleAware, load_horizon, load_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
BASE_MODEL_ID = "google/timesfm-2.5-200m-transformers"


def load_all(data_dir: Path, intervals, min_len):
  out = []
  for iv in intervals:
    for f in sorted((data_dir / iv).glob("*.parquet")):
      t, v = load_ohlcv(f)
      if len(v) >= min_len:
        out.append((f.stem, iv, t, v))
  return out


class Sampler:
  """Balanced over intervals, then series, then position (targets end before val_start)."""

  def __init__(self, series, val_start, ctx, hor, seed):
    self.ctx, self.hor = ctx, hor
    self.rng = np.random.default_rng(seed)
    self.pools: dict[str, list] = {}
    for _, iv, t, v in series:
      n = int(np.searchsorted(t, val_start))
      if n >= ctx + hor:
        self.pools.setdefault(iv, []).append(v[:n])
    self.ivs = sorted(self.pools)

  def batch(self, bs, ctx=None):
    ctx = ctx or self.ctx
    xs, ys = [], []
    for _ in range(bs):
      pool = self.pools[self.ivs[self.rng.integers(len(self.ivs))]]
      v = pool[self.rng.integers(len(pool))]
      i = self.rng.integers(ctx, len(v) - self.hor + 1)
      xs.append(v[i - ctx : i])
      ys.append(v[i : i + self.hor, 0])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def fixed_windows(series, lo, hi, ctx, hor, cap, stride):
  groups: dict[str, tuple[list, list]] = {}
  for _, iv, t, v in series:
    idx = np.arange(ctx, len(v) - hor + 1)
    m = t[idx] >= lo
    if hi is not None:
      m &= t[idx] < hi
    idx = idx[m][::stride]
    if len(idx) > cap:
      idx = idx[np.linspace(0, len(idx) - 1, cap).round().astype(int)]
    c, y = groups.setdefault(iv, ([], []))
    for i in idx:
      c.append(v[i - ctx : i])
      y.append(v[i : i + hor, 0])
  return {k: (np.stack(c), np.stack(y)) for k, (c, y) in groups.items() if c}


def forward(model, ca, x):
  """x: [B, T, 4]. Uses the candle encoder when `ca` is given."""
  with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
    if ca is None:
      return model(past_values=x[..., 0], forecast_context_len=x.shape[1]).full_predictions
    with ca.features(x):
      return model(past_values=x[..., 0], forecast_context_len=x.shape[1]).full_predictions


def rel_loss(full, x, y):
  last = x[:, -1:, 0].float()
  pred = full[:, : y.shape[1]].float() / last[..., None]
  tgt = (y.float() / last)[..., None]
  lv = torch.linspace(0.1, 0.9, 9, device=x.device)
  err = tgt - pred[..., 1:]
  return 100 * (torch.maximum(lv * err, (lv - 1) * err).mean() + (tgt[..., 0] - pred[..., 0]).abs().mean())


@torch.no_grad()
def predict(model, ca, c, hor, device, bs=256):
  out = []
  for i in range(0, len(c), bs):
    x = torch.from_numpy(c[i : i + bs]).to(device)
    out.append(forward(model, ca, x)[:, :hor].float().cpu().numpy())
  return np.concatenate(out)


def val_loss(model, ca, windows, device, contexts):
  """Mean over intervals and over the given context lengths (same forecast moments)."""
  model.eval()
  per = []
  with torch.no_grad():
    for c, y in windows.values():
      for L in contexts:
        bs = max(16, 256 * 512 // L)
        tot = 0.0
        for i in range(0, len(c), bs):
          x = torch.from_numpy(np.ascontiguousarray(c[i : i + bs, -L:])).to(device)
          t = torch.from_numpy(y[i : i + bs]).to(device)
          tot += rel_loss(forward(model, ca, x), x, t).item() * len(x)
        per.append(tot / len(c))
  model.train()
  return float(np.mean(per))


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--name", default="horizon-c3")
  p.add_argument("--init_adapter", default="horizon-c2")
  p.add_argument("--no_candle", action="store_true", help="Control run: close only")
  p.add_argument("--data_dir", default="data/crypto")
  p.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "1h"])
  p.add_argument("--val_start", default="2026-05-01")
  p.add_argument("--test_start", default="2026-07-01")
  p.add_argument("--context_len", type=int, default=512)
  p.add_argument("--context_choices", type=int, nargs="+", default=None,
                 help="Train on a random length per batch, e.g. 512 1024 2048 4096. Batch size "
                      "shrinks with length so each batch holds the same number of candles.")
  p.add_argument("--val_contexts", type=int, nargs="+", default=None,
                 help="Lengths averaged for checkpoint selection (default: shortest and longest choice)")
  p.add_argument("--horizon_len", type=int, default=24)
  p.add_argument("--steps", type=int, default=1500)
  p.add_argument("--batch_size", type=int, default=256)
  p.add_argument("--lr", type=float, default=3e-5, help="LoRA learning rate")
  p.add_argument("--enc_lr", type=float, default=3e-4, help="Candle encoder learning rate (new layer)")
  p.add_argument("--warmup", type=int, default=100)
  p.add_argument("--eval_every", type=int, default=100)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--eval_only", action="store_true")
  p.add_argument("--compare", nargs="+", default=["none", "horizon-c2"])
  args = p.parse_args()

  from peft import PeftModel
  from transformers import TimesFm2_5ModelForPrediction

  torch.manual_seed(args.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  choices = args.context_choices or [args.context_len]
  val_ctx = args.val_contexts or sorted({min(choices), max(choices)})
  ctx, hor = max(choices), args.horizon_len
  vs, ts = np.datetime64(pd.Timestamp(args.val_start)), np.datetime64(pd.Timestamp(args.test_start))
  out_dir = Path("adapters") / args.name

  series = load_all(Path(args.data_dir), args.intervals, ctx + hor)
  val_w = fixed_windows(series, vs, ts, ctx, hor, 48, hor)
  test_w = fixed_windows(series, ts, None, ctx, hor, 200, hor)
  log.info("%d series | val %s | test %s", len(series),
           {k: len(v[0]) for k, v in val_w.items()}, {k: len(v[0]) for k, v in test_w.items()})

  if not args.eval_only:
    base = TimesFm2_5ModelForPrediction.from_pretrained(BASE_MODEL_ID, dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(base, Path("adapters") / args.init_adapter, is_trainable=True)
    ca = None if args.no_candle else CandleAware.attach(model, device)
    lora_params = [q for q in model.parameters() if q.requires_grad]
    groups = [{"params": lora_params, "lr": args.lr}]
    if ca is not None:
      groups.append({"params": list(ca.encoder.parameters()), "lr": args.enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    sampler = Sampler(series, vs, ctx, hor, args.seed)

    def save():
      out_dir.mkdir(parents=True, exist_ok=True)
      model.save_pretrained(out_dir)
      if ca is not None:
        ca.save(out_dir / "candle_encoder.safetensors")

    best = val_loss(model, ca, val_w, device, val_ctx)
    log.info("val at start (= %s), contexts %s: %.4f", args.init_adapter, val_ctx, best)
    save()
    all_params = lora_params + (list(ca.encoder.parameters()) if ca else [])
    model.train()
    run, t0 = 0.0, time.time()
    for step in range(1, args.steps + 1):
      L = choices[sampler.rng.integers(len(choices))]
      x, y = sampler.batch(max(8, args.batch_size * min(choices) // L), L)
      x, y = x.to(device), y.to(device)
      loss = rel_loss(forward(model, ca, x), x, y)
      if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        continue
      loss.backward()
      torch.nn.utils.clip_grad_norm_(all_params, 1.0)
      opt.step()
      sched.step()
      opt.zero_grad(set_to_none=True)
      run += loss.item()
      if step % args.eval_every == 0 or step == args.steps:
        v = val_loss(model, ca, val_w, device, val_ctx)
        tag = ""
        if v < best:
          best, tag = v, "  <- saved"
          save()
        log.info("step %5d/%d  train %.4f  val %.4f  %.1f it/s%s", step, args.steps,
                 run / args.eval_every, v, args.eval_every / (time.time() - t0), tag)
        run, t0 = 0.0, time.time()
    meta = {k: v for k, v in vars(args).items() if k not in ("eval_only", "compare")}
    meta.update(base_model=BASE_MODEL_ID, best_val_loss=best, context_len=args.context_len,
                features=None if args.no_candle else ["up_wick", "down_wick", "rel_volume"])
    (out_dir / "training_config.json").write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()

  # ---- test ----
  names = list(dict.fromkeys(args.compare + [args.name]))
  res = {}
  for n in names:
    m, c = load_horizon(n, device)
    res[n] = {iv: metrics(predict(m, c, cw, hor, device), cw[..., 0], y) for iv, (cw, y) in sorted(test_w.items())}
    del m
    torch.cuda.empty_cache()
  print(f"\n=== TEST (unseen, from {args.test_start}), horizon {hor} ===")
  for iv in sorted(test_w):
    print(f"[{iv}] {res[names[0]][iv]['windows']} windows | naive MAE {res[names[0]][iv]['naive_MAE_%']:.3f}%")
    print(f"  {'metric':<18}" + "".join(f"{n:>13}" for n in names))
    for k in ["MAE_%", "pinball_%", "direction_acc_%", "coverage80_%"]:
      print(f"  {k:<18}" + "".join(f"{res[n][iv][k]:>13.3f}" for n in names))
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "test_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()
