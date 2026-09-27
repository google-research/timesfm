#!/usr/bin/env python3
"""Forecast with the base TimesFM 2.5 model or any domain adapter.

Examples:
    # Live Binance data, crypto adapter
    python forecast.py --adapter horizon-c1 --symbol BTCUSDT --interval 1h

    # Same thing with the untouched base model, for comparison
    python forecast.py --adapter none --symbol BTCUSDT --interval 1h

    # Side-by-side base vs adapter, saved as a chart
    python forecast.py --adapter horizon-c1 --symbol SOLUSDT --interval 1d --compare --plot sol.png

    # Any CSV with a value column (e.g. stocks later)
    python forecast.py --adapter stocks --csv aapl.csv --column close
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

BASE_MODEL_ID = "google/timesfm-2.5-200m-transformers"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def pick_device() -> torch.device:
  """CUDA on NVIDIA, MPS on Apple Silicon Macs, otherwise CPU."""
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def load_model(adapter: str, device):
  """Returns (model, adapter_config). adapter='none' gives the plain base model."""
  from transformers import TimesFm2_5ModelForPrediction

  model = TimesFm2_5ModelForPrediction.from_pretrained(BASE_MODEL_ID, dtype=torch.float32).to(device)
  cfg = {}
  if adapter != "none":
    from peft import PeftModel

    path = Path(__file__).parent / "adapters" / adapter
    if not path.exists():
      raise SystemExit(f"No adapter at {path}. Train it with: python finetune_domain.py --domain {adapter}")
    model = PeftModel.from_pretrained(model, path)
    cfg_path = path / "training_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
  return model.eval(), cfg


@torch.no_grad()
def run(model, context: np.ndarray, horizon: int, device) -> np.ndarray:
  x = torch.tensor(context, dtype=torch.float32, device=device)[None]
  with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
    out = model(past_values=x, forecast_context_len=x.shape[1])
  return out.full_predictions[0, :horizon].float().cpu().numpy()  # [h, 10]


def binance_recent(symbol: str, interval: str, n: int) -> pd.DataFrame:
  r = requests.get(
      "https://data-api.binance.vision/api/v3/klines",
      params={"symbol": symbol, "interval": interval, "limit": min(n + 1, 1000)},
      timeout=30,
  )
  r.raise_for_status()
  rows = r.json()[:-1]  # drop the still-forming candle
  return pd.DataFrame(
      {"timestamp": pd.to_datetime([k[0] for k in rows], unit="ms", utc=True),
       "close": [float(k[4]) for k in rows]}
  )


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--adapter", default="horizon-c1", help="Adapter name under adapters/, or 'none'")
  p.add_argument("--symbol", default="BTCUSDT")
  p.add_argument("--interval", default="1h")
  p.add_argument("--csv", help="Use a CSV instead of Binance (needs a timestamp column + --column)")
  p.add_argument("--column", default="close")
  p.add_argument("--context_len", type=int, default=None, help="Defaults to what the adapter was trained with (512)")
  p.add_argument("--horizon", type=int, default=None, help="Defaults to what the adapter was trained with (24)")
  p.add_argument("--compare", action="store_true", help="Also run the base model and show both")
  p.add_argument("--plot", help="Save a chart to this path")
  args = p.parse_args()

  device = pick_device()
  model, cfg = load_model(args.adapter, device)
  ctx = args.context_len or cfg.get("context_len", 512)
  hor = args.horizon or cfg.get("horizon_len", 24)

  if args.csv:
    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    label = Path(args.csv).stem
  else:
    df = binance_recent(args.symbol, args.interval, ctx)
    label = f"{args.symbol} {args.interval}"
  values = df[args.column].to_numpy(np.float32)[-ctx:]
  times = df["timestamp"].iloc[-len(values):]
  if len(values) < ctx:
    print(f"Note: only {len(values)} points available (adapter trained on {ctx}).")

  preds = {args.adapter: run(model, values, hor, device)}
  if args.compare and args.adapter != "none":
    with model.disable_adapter():
      preds["base"] = run(model, values, hor, device)

  step = times.iloc[-1] - times.iloc[-2]
  future = pd.date_range(times.iloc[-1] + step, periods=hor, freq=step)
  last = values[-1]
  print(f"\n{label}   last close {last:,.6g} at {times.iloc[-1]}   adapter: {args.adapter}")
  table = pd.DataFrame({"time": future})
  for name, f in preds.items():
    table[f"{name}_median"] = f[:, 5]
    table[f"{name}_p10"] = f[:, 1]
    table[f"{name}_p90"] = f[:, 9]
  with pd.option_context("display.float_format", "{:,.6g}".format, "display.width", 200):
    print(table.to_string(index=False))
  for name, f in preds.items():
    chg = 100 * (f[-1, 5] / last - 1)
    print(f"{name:>8}: median {chg:+.2f}% by {future[-1]}, 80% band "
          f"[{100 * (f[-1, 1] / last - 1):+.2f}%, {100 * (f[-1, 9] / last - 1):+.2f}%]")

  if args.plot:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    show = min(len(values), hor * 5)
    ax.plot(times.iloc[-show:], values[-show:], color="#444", lw=1.2, label="history")
    colors = {"base": "#8a8a8a"}
    for name, f in preds.items():
      c = colors.get(name, "#d97706")
      ax.plot(future, f[:, 5], color=c, lw=2, label=f"{name} median")
      ax.fill_between(future, f[:, 1], f[:, 9], color=c, alpha=0.18, label=f"{name} 10-90%")
    ax.set_title(label)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"Chart saved to {args.plot}")


if __name__ == "__main__":
  main()
