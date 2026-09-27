# Domain adapters for TimesFM 2.5

One untouched base model (`google/timesfm-2.5-200m-transformers`) plus small
LoRA adapters (~11 MB each) that specialise it for a domain:

| Adapter   | Data                                       | Status  |
|-----------|--------------------------------------------|---------|
| `none`    | the original base model                    | always available |
| `horizon-c1` | Horizon-C1 (crypto): Binance spot candles, 36 USDT pairs, 1h + 1d | trained, weights in `adapters/horizon-c1/` |
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
