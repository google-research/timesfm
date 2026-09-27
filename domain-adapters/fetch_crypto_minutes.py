#!/usr/bin/env python3
"""Download 1-minute crypto candles from Binance's bulk monthly archives,
then build 5m and 15m candles from them.

Much faster than the REST API for minute data (one zip per coin per month).
The current, not-yet-archived month is topped up afterwards with:
    python fetch_crypto.py --intervals 1m --symbols <same symbols>

Output: data/crypto/{1m,5m,15m}/<SYMBOL>.parquet (same layout as fetch_crypto.py)

Usage:
    python fetch_crypto_minutes.py                           # top 10 coins, last 24 months
    python fetch_crypto_minutes.py --months 36 --symbols BTCUSDT ETHUSDT
    python fetch_crypto_minutes.py --resample_only           # rebuild 5m/15m from 1m
"""

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{ym}.zip"
TOP10 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def fetch_month(symbol: str, ym: str) -> pd.DataFrame | None:
  url = ARCHIVE.format(s=symbol, ym=ym)
  for attempt in range(4):
    try:
      r = requests.get(url, timeout=120)
      if r.status_code == 404:
        return None
      r.raise_for_status()
      break
    except requests.RequestException:
      if attempt == 3:
        raise
  with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
  if not str(df.iloc[0, 0]).isdigit():  # some archives carry a header row
    df = df.iloc[1:].astype({"open_time": "int64"})
  t = df["open_time"].astype("int64")
  # Binance switched spot archives from milliseconds to microseconds in 2025.
  unit = "us" if t.iloc[0] > 10**14 else "ms"
  out = pd.DataFrame({"timestamp": pd.to_datetime(t, unit=unit, utc=True)})
  for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
    out[c] = df[c].astype("float64").to_numpy()
  out["trades"] = df["trades"].astype("int64").to_numpy()
  return out


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
  g = df.set_index("timestamp").resample(rule, label="left", closed="left")
  out = pd.DataFrame({
      "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
      "close": g["close"].last(), "volume": g["volume"].sum(),
      "quote_volume": g["quote_volume"].sum(), "trades": g["trades"].sum(),
  }).dropna(subset=["close"])
  return out.reset_index()


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--symbols", nargs="+", default=TOP10)
  p.add_argument("--months", type=int, default=24, help="Number of full past months to fetch")
  p.add_argument("--out_dir", default="data/crypto")
  p.add_argument("--resample_only", action="store_true")
  args = p.parse_args()

  root = Path(args.out_dir)
  (root / "1m").mkdir(parents=True, exist_ok=True)
  last_full = pd.Timestamp.now(tz="UTC").to_period("M") - 1
  months = [str(last_full - i) for i in range(args.months)][::-1]

  for symbol in args.symbols:
    path = root / "1m" / f"{symbol}.parquet"
    if not args.resample_only:
      with ThreadPoolExecutor(max_workers=6) as pool:
        parts = [d for d in pool.map(lambda ym: fetch_month(symbol, ym), months) if d is not None]
      if not parts:
        print(f"{symbol}: no archives found", flush=True)
        continue
      df = pd.concat(parts, ignore_index=True)
      if path.exists():  # keep anything newer that the REST top-up already added
        df = pd.concat([df, pd.read_parquet(path)], ignore_index=True)
      df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
      df.to_parquet(path, index=False)
    df = pd.read_parquet(path)
    for rule, name in (("5min", "5m"), ("15min", "15m")):
      (root / name).mkdir(parents=True, exist_ok=True)
      resample(df, rule).to_parquet(root / name / f"{symbol}.parquet", index=False)
    print(f"{symbol:>9}: {len(df):>9,} 1m candles  {df['timestamp'].iloc[0]:%Y-%m-%d} -> "
          f"{df['timestamp'].iloc[-1]:%Y-%m-%d %H:%M}", flush=True)


if __name__ == "__main__":
  main()
