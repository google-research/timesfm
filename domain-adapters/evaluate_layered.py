#!/usr/bin/env python3
"""Replay the "layered" live routine on unseen history and score each model.

Routine being simulated, e.g. for a 15-minute candle with 1-minute updates:
  minute 0:  forecast the whole candle -> call up/down (close vs candle open)
  minute 1:  one 1m bar of the candle is known; re-forecast the remaining 14
  ...
  minute 14: re-forecast the last minute

At every update minute we report the direction accuracy of each model AND of a
no-model baseline: "is the price right now above or below the candle open?".
Late in the candle most of the move has already happened, so both numbers
climb. The model only adds value where it beats the baseline.

Usage:
    python evaluate_layered.py --candle 15 --step 1m            # 15m candle, 1m updates
    python evaluate_layered.py --candle 60 --step 5m            # 1h candle, 5m updates
    python evaluate_layered.py --adapters none horizon-c1 horizon-c2
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BASE_MODEL_ID = "google/timesfm-2.5-200m-transformers"
TOP10 = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
         "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT"]
STEP_MIN = {"1m": 1, "5m": 5, "15m": 15}


def build_candles(path: Path, candle_min: int, step_min: int, ctx: int, test_start, max_candles, rng):
  """Returns (closes [N, ctx + L], candle_open [N]) for aligned, gap-free candles."""
  df = pd.read_parquet(path, columns=["timestamp", "open", "close"])
  t = df["timestamp"].to_numpy("datetime64[m]").astype("int64")  # minutes since epoch
  close, open_ = df["close"].to_numpy(np.float32), df["open"].to_numpy(np.float32)
  L = candle_min // step_min
  start_min = pd.Timestamp(test_start).value // 60_000_000_000

  idx = np.where((t % candle_min == 0) & (t >= start_min))[0]
  idx = idx[(idx >= ctx) & (idx + L <= len(t))]
  # The candle itself and the context must be gap-free (exchange outages happen).
  idx = idx[t[idx + L - 1] - t[idx] == (L - 1) * step_min]
  idx = idx[t[idx - 1] - t[idx - ctx] == (ctx - 1) * step_min]
  if len(idx) > max_candles:
    idx = np.sort(rng.choice(idx, max_candles, replace=False))
  windows = np.stack([close[i - ctx : i + L] for i in idx]) if len(idx) else np.zeros((0, ctx + L), np.float32)
  return windows, open_[idx]


@torch.no_grad()
def predict_close(model, contexts: np.ndarray, steps_ahead: int, device, bs: int = 512):
  """Point forecast and quantiles of the value `steps_ahead` bars after the context."""
  pts, qs = [], []
  for i in range(0, len(contexts), bs):
    x = torch.from_numpy(contexts[i : i + bs]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
      f = model(past_values=x, forecast_context_len=x.shape[1]).full_predictions
    f = f[:, steps_ahead - 1].float().cpu().numpy()  # [b, 10]
    pts.append(f[:, 0])
    qs.append(f[:, 1:])
  return np.concatenate(pts), np.concatenate(qs)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--adapters", nargs="+", default=["none", "horizon-c1", "horizon-c2"])
  p.add_argument("--candle", type=int, default=15, help="Candle length in minutes")
  p.add_argument("--step", default="1m", choices=list(STEP_MIN), help="Update / data interval")
  p.add_argument("--symbols", nargs="+", default=TOP10)
  p.add_argument("--data_dir", default="data/crypto")
  p.add_argument("--test_start", default="2026-07-01")
  p.add_argument("--context_len", type=int, default=512)
  p.add_argument("--max_candles", type=int, default=1500, help="Per coin, sampled at random")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  from peft import PeftModel
  from transformers import TimesFm2_5ModelForPrediction

  step_min = STEP_MIN[args.step]
  L = args.candle // step_min
  assert args.candle % step_min == 0 and L >= 2
  ctx = args.context_len
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  rng = np.random.default_rng(args.seed)

  wins, opens = [], []
  for s in args.symbols:
    w, o = build_candles(Path(args.data_dir) / args.step / f"{s}.parquet", args.candle, step_min,
                         ctx, args.test_start, args.max_candles, rng)
    wins.append(w)
    opens.append(o)
  W, O = np.concatenate(wins), np.concatenate(opens)
  final_close = W[:, ctx + L - 1]
  truth = np.sign(final_close - O)
  keep = truth != 0  # flat candles have no direction
  W, O, truth = W[keep], O[keep], truth[keep]
  log.info("%d candles of %d min (%d x %s updates) since %s, %d coins",
           len(W), args.candle, L, args.step, args.test_start, len(args.symbols))

  base = TimesFm2_5ModelForPrediction.from_pretrained(BASE_MODEL_ID, dtype=torch.float32).to(device)
  model = None
  for a in args.adapters:
    if a == "none":
      continue
    path = Path(__file__).parent / "adapters" / a
    if model is None:
      model = PeftModel.from_pretrained(base, path, adapter_name=a)
    else:
      model.load_adapter(path, adapter_name=a)
  model = (model or base).eval()

  # results[name][k] = accuracy after k bars of the candle are known
  results: dict[str, dict] = {"baseline (price vs open)": {}}
  for a in args.adapters:
    results[a] = {}
    results[f"{a} | confident calls"] = {}
    results[f"{a} | when disagreeing w/ baseline"] = {}

  for k in range(L):
    # Context ends at the last completed bar: k bars into the candle.
    contexts = W[:, k : k + ctx]
    current = contexts[:, -1]
    base_call = np.sign(current - O)
    # Ties (price exactly at open, always at k=0) count as a coin flip.
    base_acc = np.where(base_call == 0, 0.5, base_call == truth).mean()
    results["baseline (price vs open)"][k] = base_acc

    for a in args.adapters:
      if a == "none":
        ctxmgr = model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad()
      else:
        model.set_adapter(a)
        ctxmgr = torch.no_grad()
      with ctxmgr:
        pt, q = predict_close(model, np.ascontiguousarray(contexts), L - k, device)
      call = np.sign(pt - O)
      results[a][k] = np.where(call == 0, 0.5, call == truth).mean()
      # Confident: the model's 20%..80% range sits entirely on one side of the open.
      conf = (q[:, 1] > O) | (q[:, 7] < O)
      results[f"{a} | confident calls"][k] = (
          (call[conf] == truth[conf]).mean() if conf.any() else np.nan, conf.mean())
      dis = (call != base_call) & (base_call != 0)
      results[f"{a} | when disagreeing w/ baseline"][k] = (
          (call[dis] == truth[dis]).mean() if dis.any() else np.nan, dis.mean())
    log.info("update %2d/%d done", k + 1, L)

  # ---- report ----
  print(f"\nDirection accuracy, {args.candle}-min candle, re-forecast every {args.step}"
        f"  ({len(W)} candles since {args.test_start})")
  head = f"{'bars known':>10} | {'baseline':>8} | " + " | ".join(f"{a:>10}" for a in args.adapters)
  print(head)
  print("-" * len(head))
  for k in range(L):
    row = f"{k:>6}/{L:<3} | {100 * results['baseline (price vs open)'][k]:7.1f}% | "
    row += " | ".join(f"{100 * results[a][k]:9.1f}%" for a in args.adapters)
    print(row)

  for a in args.adapters:
    print(f"\n{a}: when it DISAGREES with the baseline, how often is the model right?  (share of candles)")
    print("   " + "  ".join(
        f"k={k}: {100 * v[0]:.0f}% ({100 * v[1]:.0f}%)" for k, v in results[f"{a} | when disagreeing w/ baseline"].items()
        if k in (1, L // 3, 2 * L // 3, L - 1) and not np.isnan(v[0])))
    print(f"{a}: confident calls (20-80% range fully above/below open): accuracy (share of candles)")
    print("   " + "  ".join(
        f"k={k}: {100 * v[0]:.0f}% ({100 * v[1]:.0f}%)" for k, v in results[f"{a} | confident calls"].items()
        if k in (0, 1, L // 3, 2 * L // 3, L - 1) and not np.isnan(v[0])))

  out = Path("runs") / f"layered_{args.candle}m_{args.step}.json"
  out.parent.mkdir(exist_ok=True)
  out.write_text(json.dumps({k: {str(kk): vv for kk, vv in v.items()} for k, v in results.items()},
                            indent=1, default=float))
  print(f"\nSaved {out}")


if __name__ == "__main__":
  main()
