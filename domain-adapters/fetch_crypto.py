#!/usr/bin/env python3
"""Download historical crypto OHLCV candles from Binance's public market-data API.

No API key or account needed. Saves one parquet per (symbol, interval) to
data/crypto/<interval>/<SYMBOL>.parquet with columns:
    timestamp (UTC), open, high, low, close, volume, quote_volume, trades

Re-running is incremental: only candles after the last saved timestamp are fetched.

Usage:
    python fetch_crypto.py                         # default symbols, 1h + 1d
    python fetch_crypto.py --intervals 1h 4h 1d
    python fetch_crypto.py --symbols BTCUSDT ETHUSDT --intervals 15m
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data-api.binance.vision/api/v3/klines"

# Liquid USDT pairs with long histories. Stablecoins are deliberately excluded -
# they are flat at ~1.0 and would teach the model nothing useful.
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT",
    "BCHUSDT", "XLMUSDT", "ATOMUSDT", "ETCUSDT", "NEARUSDT", "UNIUSDT",
    "FILUSDT", "AAVEUSDT", "ALGOUSDT", "VETUSDT", "HBARUSDT", "ICPUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "SHIBUSDT",
    "PEPEUSDT", "SEIUSDT", "TIAUSDT", "RNDRUSDT", "FETUSDT", "MKRUSDT",
]

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]

INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "12h": 43_200_000, "1d": 86_400_000, "1w": 604_800_000,
}


def fetch_klines(session: requests.Session, symbol: str, interval: str, start_ms: int) -> list:
  """Fetch all closed candles for symbol/interval starting at start_ms."""
  rows: list = []
  now_ms = int(time.time() * 1000)
  while True:
    for attempt in range(5):
      try:
        r = session.get(
            BASE_URL,
            params={"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000},
            timeout=30,
        )
        if r.status_code == 429 or r.status_code == 418:
          time.sleep(int(r.headers.get("Retry-After", 30)))
          continue
        if r.status_code == 400:  # unknown / delisted symbol
          return rows
        r.raise_for_status()
        batch = r.json()
        break
      except requests.RequestException as e:
        wait = 2 ** attempt
        print(f"  {symbol} {interval}: {e} - retrying in {wait}s")
        time.sleep(wait)
    else:
      raise RuntimeError(f"Failed to fetch {symbol} {interval} after retries")

    if not batch:
      break
    # Drop the still-open candle so we never store a partial bar.
    batch = [k for k in batch if k[6] < now_ms]
    rows.extend(batch)
    if len(batch) < 1000:
      break
    start_ms = batch[-1][0] + INTERVAL_MS[interval]
    time.sleep(0.1)
  return rows


def to_frame(rows: list) -> pd.DataFrame:
  df = pd.DataFrame(rows, columns=COLUMNS)
  df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
  for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
    df[c] = df[c].astype("float64")
  df["trades"] = df["trades"].astype("int64")
  return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
  p.add_argument("--intervals", nargs="+", default=["1h", "1d"])
  p.add_argument("--start", default="2017-01-01", help="Earliest date to fetch (UTC)")
  p.add_argument("--out_dir", default="data/crypto")
  args = p.parse_args()

  session = requests.Session()
  start_default = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)

  for interval in args.intervals:
    out = Path(args.out_dir) / interval
    out.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
      path = out / f"{symbol}.parquet"
      existing = pd.read_parquet(path) if path.exists() else None
      start_ms = start_default
      if existing is not None and len(existing):
        start_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000) + INTERVAL_MS[interval]

      rows = fetch_klines(session, symbol, interval, start_ms)
      if not rows and existing is None:
        print(f"{symbol:>10} {interval:>3}: no data (not listed?) - skipped")
        continue
      new = to_frame(rows) if rows else None
      df = pd.concat([d for d in (existing, new) if d is not None], ignore_index=True)
      df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
      df.to_parquet(path, index=False)
      print(
          f"{symbol:>10} {interval:>3}: {len(df):>7,} candles "
          f"{df['timestamp'].iloc[0]:%Y-%m-%d} -> {df['timestamp'].iloc[-1]:%Y-%m-%d %H:%M} "
          f"(+{len(rows):,} new)",
      flush=True)


if __name__ == "__main__":
  main()
