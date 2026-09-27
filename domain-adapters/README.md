# Domain adapters for TimesFM 2.5

One untouched base model (`google/timesfm-2.5-200m-transformers`) plus small
LoRA adapters (~11 MB each) that specialise it for a domain:

| Adapter   | Data                                       | Status  |
|-----------|--------------------------------------------|---------|
| `none`    | the original base model                    | always available |
| `horizon-c1` | Horizon-C1 (crypto): Binance spot candles, 36 USDT pairs, 1h + 1d | trained, weights in `adapters/horizon-c1/` |
| `horizon-c2` | Horizon-C2 (crypto intraday): 1m/5m/15m for 10 top pairs + 1h for 36 pairs | trained, weights in `adapters/horizon-c2/` |
| `stocks`  | TBD, same pipeline, different fetcher       | planned |

## Setup (Windows, RTX 50-series)

Blackwell GPUs need PyTorch built for CUDA 12.8+:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe transformers accelerate peft pandas pyarrow requests matplotlib safetensors
```

## Crypto workflow

```powershell
.venv\Scripts\python.exe fetch_crypto.py                     # ~20 min first time; incremental after
.venv\Scripts\python.exe finetune_domain.py --domain crypto  # ~8 min on a 5070 Ti
.venv\Scripts\python.exe forecast.py --adapter horizon-c1 --symbol BTCUSDT --interval 1h --compare --plot btc.png
```

## How the evaluation stays honest

All assets share the same calendar cutoffs (crypto moves together, so a random
split would leak BTC's future into ETH's test):

- **train**: everything before `--val_start` (default 2025-09-01)
- **val**: 2025-09-01 to 2026-03-01, used only to pick the best checkpoint
- **test**: 2026-03-01 onward, never seen, reported at the end

The report compares base vs adapter on:
- `MAE_%`: average absolute error as % of the last price
- `skill_vs_naive_%`: improvement over "price stays flat" (positive means it beats naive)
- `pinball_%`: quality of the whole forecast distribution (lower is better)
- `direction_acc_%`: whether up/down at the end of the horizon was right
- `coverage80_%`: how often the truth landed in the 10-90% band (ideal: 80)

## Adding stocks later

Write a fetcher that produces `data/stocks/<interval>/<TICKER>.parquet` with
`timestamp` and `close` columns, then:

```powershell
.venv\Scripts\python.exe finetune_domain.py --domain stocks --intervals 1d
```

## Notes

- Don't normalise prices yourself. TimesFM normalises each window internally,
  and the default `relative` loss measures error as % of the last price, so
  BTC at $80k and SHIB at $0.00001 contribute equally.
- Training step time is overhead-bound on Windows, so batch size 256 costs
  about the same per step as 64. Drop it if you hit out-of-memory errors.
- Train and forecast use context length 512 by default. Other lengths still
  work, but the adapter is best at the length it was trained with.

## Crypto results (test period 2026-03-01 to 2026-09-26, never trained on)

| | base | Horizon-C1 | naive (flat) |
|---|---|---|---|
| 1d MAE, 24-day horizon | 9.91% | **9.24%** | 9.25% |
| 1h MAE, 24-hour horizon | 2.15% | **2.09%** | 1.76% |
| 1d pinball (distribution) | 3.79 | 3.79 | – |
| 1h pinball (distribution) | 0.893 | **0.876** | – |
| Direction accuracy | 51-54% | 50-54% | – |

Most of the gain comes from the adapter learning to be **less trend-happy**: its
median predicted 24-step move is about half the base model's. Neither model
predicts direction better than a coin flip, which is expected from price history alone.

Lessons from the first attempt (`adapters/crypto_v1_native_loss`, kept for reference):
- The built-in loss rewarded over-extrapolating trends, which made daily error
  **2x worse** even though the validation loss looked better. The default loss is
  now `--loss relative` (error as % of the last price).
- The best checkpoint arrives early (~200 steps at lr 5e-5), and longer training
  overfits the 2017-2025 regime. That's why the default is 800 steps.

Range (80% band) on the same test period: Horizon-C1's bands are 6-11% narrower,
with a slightly lower hit rate (1d 77% vs 83%, 1h 73% vs 75%). The combined
interval score is only 1-2% better, so the range improvement is small.

## Running on a Mac (Apple Silicon)

`forecast.py` picks MPS automatically. Install with the default PyTorch wheel:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate peft pandas pyarrow requests matplotlib safetensors
python forecast.py --adapter horizon-c1 --symbol BTCUSDT --interval 1h --compare
```

## Horizon-C2 (intraday)

```powershell
.venv\Scripts\python.exe fetch_crypto_minutes.py      # 1m archives -> also builds 5m, 15m
.venv\Scripts\python.exe fetch_crypto.py --intervals 1m --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT TRXUSDT AVAXUSDT LINKUSDT
.venv\Scripts\python.exe fetch_crypto_minutes.py --resample_only
.venv\Scripts\python.exe finetune_domain.py --domain horizon-c2 --data_dir data/crypto --intervals 1m 5m 15m 1h --val_start 2026-05-01 --test_start 2026-07-01 --steps 600
```

Test period 2026-07-01 to 2026-09-27, 24-step horizon, error as % of last price:

| interval | base MAE | C2 MAE | naive MAE | base pinball | C2 pinball |
|---|---|---|---|---|---|
| 1m  | 0.162 | **0.156** | 0.157 | 0.064 | **0.062** |
| 5m  | 0.389 | **0.369** | 0.369 | 0.154 | **0.149** |
| 15m | 0.692 | **0.653** | 0.648 | 0.275 | **0.265** |
| 1h  | 2.141 | **2.027** | 1.748 | 0.891 | **0.852** |

### Layered routine (`evaluate_layered.py`)

This replays "forecast the candle, then re-forecast every step" on ~14.5k candles
per setup, and compares against a no-model baseline: *is the price right now above the open?*

15-min candle, re-forecast every minute (direction accuracy):

| minutes known | baseline | base | C1 | C2 |
|---|---|---|---|---|
| 0  | 50.0% | 51.9% | 51.1% | 52.4% |
| 2  | 63.6% | 61.7% | 63.2% | 63.9% |
| 5  | 71.2% | 70.8% | 71.5% | 71.7% |
| 7  | 76.5% | 75.9% | 76.4% | 76.6% |
| 10 | 82.3% | 82.1% | 82.6% | 82.7% |
| 14 | 93.4% | 93.3% | 93.4% | 93.6% |

The accuracy that climbs through the candle comes from the candle already being
partly finished, not from the model. C2 is the only model that never does worse
than the baseline, but its edge is only 0.1-0.5 points. 1h candles with 5m updates
show the same pattern.

## Experiment: Horizon-C3 (reads high / low / volume) - no gain, not promoted

`ohlcv.py` adds a small candle encoder to TimesFM's input layer. It starts at
zero, so C3 is identical to C2 at step 0. The encoder sees, per time step, the
upper wick, the lower wick, and volume relative to its recent average.
`finetune_ohlcv.py` trains it on top of C2, and `--no_candle` runs the same
training with close prices only (`c2-control`), so extra training isn't
mistaken for a feature gain.

Test period from 2026-07-01, MAE as % of price:

| interval | C2 | control (close only) | C3 (full candle) |
|---|---|---|---|
| 1m  | 0.156 | 0.156 | 0.156 |
| 5m  | 0.369 | 0.370 | 0.369 |
| 15m | 0.653 | 0.654 | 0.654 |
| 1h  | 2.027 | 2.024 | 2.024 |

In the 15-min layered routine, all three are within 0.2 points of each other
at every minute. The encoder does change forecasts (by ~25% of the typical
predicted move), but not in a way that improves accuracy. Wick and volume
history adds nothing over the close series for these horizons.
**Horizon-C2 remains the recommended model.** The C3 code is kept so the
experiment can be rerun with other inputs.

## Experiment: Horizon-C2.1 (trained on 512-4096 candles of history): no gain, not promoted

C2 was fine-tuned further with a random history length per batch
(`finetune_ohlcv.py --no_candle --context_choices 512 1024 2048 4096`).
Test error (% of price) with 512 vs 4096 candles of history:

| interval | C2 @512 | C2.1 @512 | C2 @4096 | C2.1 @4096 |
|---|---|---|---|---|
| 1m  | 0.1573 | 0.1574 | 0.1579 | 0.1577 |
| 5m  | 0.3808 | 0.3808 | 0.3826 | 0.3813 |
| 15m | 0.6905 | 0.6906 | 0.6932 | 0.6945 |
| 1h  | 1.2769 | 1.2780 | 1.2860 | 1.2866 |

The 15-min layered routine was identical within 0.3 points. **Use C2 with
512 candles of history.** More history (or training for more) doesn't help.

Speed on an RTX 5070 Ti: ~0.20 s per call at 512 candles and ~0.47 s at 4096.
One call with 10 coins takes the same time as one call with 1 coin, so batch
your coins into a single call.
