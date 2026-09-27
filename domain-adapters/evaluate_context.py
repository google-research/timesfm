#!/usr/bin/env python3
"""Does feeding the model more history help? Scores the same forecast moments
with different context lengths (how many past candles the model sees).

    python evaluate_context.py                                   # base vs horizon-c2
    python evaluate_context.py --contexts 256 512 2048 8192 --intervals 1m 1h

Every context length forecasts the exact same moments, so differences come
only from how much history the model saw.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune_domain import metrics
from ohlcv import load_horizon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

STEP = {"1m": "1 min", "5m": "5 min", "15m": "15 min", "1h": "1 hour", "1d": "1 day"}


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--adapters", nargs="+", default=["none", "horizon-c2"])
  p.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "1h"])
  p.add_argument("--contexts", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
  p.add_argument("--data_dir", default="data/crypto")
  p.add_argument("--test_start", default="2026-07-01")
  p.add_argument("--horizon", type=int, default=24)
  p.add_argument("--per_series", type=int, default=150)
  p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                                                   "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT"])
  args = p.parse_args()
  assert all(c % 32 == 0 for c in args.contexts)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  cmax, hor = max(args.contexts), args.horizon
  ts = np.datetime64(pd.Timestamp(args.test_start))

  windows = {}
  for iv in args.intervals:
    c, y = [], []
    for s in args.symbols:
      f = Path(args.data_dir) / iv / f"{s}.parquet"
      if not f.exists():
        continue
      df = pd.read_parquet(f, columns=["timestamp", "close"])
      t = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
      v = df["close"].to_numpy(np.float32)
      idx = np.arange(cmax, len(v) - hor + 1)
      idx = idx[t[idx] >= ts][::hor]
      if len(idx) > args.per_series:
        idx = idx[np.linspace(0, len(idx) - 1, args.per_series).round().astype(int)]
      c += [v[i - cmax : i] for i in idx]
      y += [v[i : i + hor] for i in idx]
    windows[iv] = (np.stack(c), np.stack(y))
    log.info("%s: %d forecast moments", iv, len(c))

  rows = []
  for a in args.adapters:
    model, _ = load_horizon(a, device)
    for iv, (c, y) in windows.items():
      for L in args.contexts:
        bs = max(8, 256 * 512 // L)
        out = []
        with torch.no_grad():
          for i in range(0, len(c), bs):
            x = torch.from_numpy(np.ascontiguousarray(c[i : i + bs, -L:])).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
              out.append(model(past_values=x, forecast_context_len=L).full_predictions[:, :hor].float().cpu().numpy())
        m = metrics(np.concatenate(out), c[:, -L:], y)
        rows.append({"model": a, "interval": iv, "context": L, "MAE_%": m["MAE_%"], "pinball_%": m["pinball_%"],
                     "naive_MAE_%": m["naive_MAE_%"]})
      log.info("%s %s done", a, iv)
    del model
    torch.cuda.empty_cache()

  df = pd.DataFrame(rows)
  for iv in args.intervals:
    sub = df[df.interval == iv]
    print(f"\n[{iv}] error as % of price, {len(windows[iv][0])} forecasts, horizon {hor} "
          f"(naive MAE {sub['naive_MAE_%'].iloc[0]:.3f}%)")
    print(f"  {'context':>8} {'= history':>14} | " + " | ".join(f"{a:>10} MAE  pinball" for a in args.adapters))
    for L in args.contexts:
      span = pd.Timedelta(STEP[iv]) * L
      span_s = f"{span.days}d {span.seconds // 3600}h" if span.days else f"{span.seconds // 3600}h {span.seconds % 3600 // 60}m"
      cells = []
      for a in args.adapters:
        r = sub[(sub.model == a) & (sub.context == L)].iloc[0]
        cells.append(f"{r['MAE_%']:>15.4f}  {r['pinball_%']:.4f}")
      print(f"  {L:>8} {span_s:>14} | " + " | ".join(cells))
  Path("runs").mkdir(exist_ok=True)
  df.to_csv("runs/context_length.csv", index=False)


if __name__ == "__main__":
  main()
