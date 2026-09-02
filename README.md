# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   <span style="color:red">(NEW!)</span> TimesFM 3.0 Checkpoint:
    [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch).
*   Checkpoints (up to 2.5):
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
    (New blog post for TimesFM 3.0 coming soon!).
*   TimesFM in Google 1P Products:
    *   [BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model):
        Enterprise level SQL queries for scalability and reliability.
    *   [Google Sheets](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html):
        For your daily spreadsheet.
    *   [Vertex Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/timesfm):
        Dockerized endpoint for agentic calling.

This open version is not an officially supported Google product.

**Latest Model Version:** TimesFM 3.0

**Archived Model Versions:**

-   2.5: relevant code under `src/timesfm`.
-   1.0 and 2.0: relevant code archived in the subdirectory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

--------------------------------------------------------------------------------

## Update — August 2026

**TimesFM 3.0 is out!**

TimesFM 3.0 introduces native **multivariate time-series forecasting**, flexible
**covariate support** (both past-only and past-and-future covariates), superior
zero-shot generalist capabilities, and top performance across all three major
time-series foundation model benchmarks.

### Key Highlights:

-   **Native Multivariate & Univariate Forecasting with Covariates**: Seamlessly
    forecast multi-channel multivariate series as well as individual univariate
    series, with native support for past-only and past-and-future dynamic
    covariates without per-task tuning.
-   **Top Benchmark Performance**:
    -   🥇 **fev-bench**: **Rank #1 overall** across 100 diverse real-world
        forecasting tasks.
    -   🥇 **TIME Benchmark**: **Rank #1 overall** across 50 domain datasets and
        98 evaluation tasks.
    -   🥇 **GIFT-Eval**: **Rank #1 among all foundation models**.

### License notice for pretrained weights

> **Important:** The TimesFM source code in this repository is licensed under
> Apache-2.0, and model weights up to version 2.5 remain Apache-2.0. However,
> for the time being, TimesFM 3.0 pretrained weights are distributed under the
> separate `timesfm-non-commercial-license-v1.0` license and are restricted to
> non-commercial, non-production use. Commercial or production use of the
> default pretrained weights is **not permitted**.

--------------------------------------------------------------------------------

## Update - July 2, 2026

Updated PyPI to `timesfm=2.0.2`. See
[Install](https://github.com/google-research/timesfm#from-pypi).

## Update - Apr. 9, 2026

Added fine-tuning example using HuggingFace Transformers + PEFT (LoRA) — see
[`timesfm-forecasting/examples/finetuning/`](timesfm-forecasting/examples/finetuning/).
Also added unit tests (`tests/`) and incorporated several community fixes.

Shoutout to [@kashif](https://github.com/kashif) and
[@darkpowerxo](https://github.com/darkpowerxo).

## Update - Mar. 19, 2026

Huge shoutout to [@borealBytes](https://github.com/borealBytes) for adding the
support for
[AGENTS](https://github.com/google-research/timesfm/blob/master/AGENTS.md)!
TimesFM
[SKILL.md](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting)
is out.

## Update - Oct. 29, 2025

Added back the covariate support through XReg for TimesFM 2.5.

## Update - Sept. 15, 2025

TimesFM 2.5 is out!

Comparing to TimesFM 2.0, this new 2.5 model:

-   uses 200M parameters, down from 500M.
-   supports up to 16k context length, up from 2048.
-   supports continuous quantile forecast up to 1k horizon via an optional 30M
    quantile head.
-   gets rid of the `frequency` indicator.
-   has a couple of new forecasting flags.

Since the Sept. 2025 launch, the following improvements have been completed for
TimesFM 2.5:

1.  ✅ Flax version of the model for faster inference.
2.  ✅ Covariate support via XReg (see Oct. 2025 update).
3.  ✅ Documentation, examples, and agent skill (see `timesfm-forecasting/`).
4.  ✅ Fine-tuning example with LoRA via HuggingFace Transformers + PEFT (see
    `timesfm-forecasting/examples/finetuning/`).
5.  ✅ Unit tests for core layers, configs, and utilities (see `tests/`).

### Install

#### From `PyPI`

```shell
# Install TimesFM with PyTorch
pip install timesfm[torch]

# Or, for MLX-native inference on Apple silicon (no PyTorch required)
pip install timesfm[mlx]
```

#### Local Install

1.  Clone the repository:

    ```shell
    git clone https://github.com/google-research/timesfm.git
    cd timesfm
    ```

2.  Create a virtual environment and install with PyTorch:

    ```shell
    # Using uv
    uv venv
    source .venv/bin/activate

     # Install the package in editable mode with torch
    uv pip install -e .[torch]
    ```

--------------------------------------------------------------------------------

### Code Examples: TimesFM 3.0

#### 1. Univariate Forecasting (Variable Lengths)

Pass a batch of 1D NumPy arrays of different context lengths to forecast
univariate time series:

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

# Initialize TimesFM 3.0
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=32,
    device="cuda"
)
forecaster = TimesFM3Evaluator(config)

# Two univariate series of different lengths (100 and 72 steps)
ts1 = np.linspace(0, 1, 100).astype(np.float32)
ts2 = np.sin(np.linspace(0, 24, 72)).astype(np.float32)

# Generate forecast (point predictions + 9 quantiles: 0.1 to 0.9)
outputs = list(forecaster.predict_batch([ts1, ts2], horizon=12, return_quantiles=True, use_symmetric_averaging=False))

print("Series 1 forecast shape:", outputs[0].forecast.shape)   # (12,)
print("Series 1 quantiles shape:", outputs[0].quantiles.shape) # (12, 9)

print("Series 2 forecast shape:", outputs[1].forecast.shape)   # (12,)
print("Series 2 quantiles shape:", outputs[1].quantiles.shape) # (12, 9)
```

#### Apple Silicon: MLX backend

An MLX-native backend runs TimesFM 3.0 on Apple silicon without PyTorch. It mirrors the PyTorch
`TimesFM3Forecaster` interface and is numerically matched to it (median forecast max abs error
`9.5e-7`, quantiles `1.8e-6`, on `google/timesfm-3.0-pytorch`, context 512 → horizon 64).

```python
import numpy as np
from timesfm3.mlx import TimesFM3Forecaster

forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

context = np.sin(np.linspace(0, 40, 512)).astype(np.float32)
out = forecaster.predict(context, horizon=64, return_quantiles=True)
print(out.forecast.shape)    # (64,)      median forecast
print(out.quantiles.shape)   # (64, 9)    9 deciles

# batch many series through one forward pass
outs = list(forecaster.predict_batch([context] * 32, horizon=64))
```

Benchmarks (330M model, Apple M4 Max, context 512, horizon 64, fp32 with `mx.compile`):

| batch | p50 latency | throughput |
|------:|------------:|-----------:|
|     1 |     12.4 ms |   80 series/s |
|     8 |     22.0 ms |  363 series/s |
|    32 |     55.0 ms |  582 series/s |

Not yet supported by the MLX backend: covariates (`past_only_covariates` /
`past_future_covariates`), `use_symmetric_averaging`, `use_znorm`, and non-`"none"` `padding_mode`
— use the PyTorch backend for those.

#### 2. Multivariate Forecasting with Covariates

Pass a 2D array of shape `(num_variates, context_length)` along with optional
past-only and past-and-future covariates:

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

# Initialize TimesFM 3.0
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=16,
    device="cuda"
)
forecaster = TimesFM3Evaluator(config)

context_len = 128
horizon = 24

# 3 target variates across past context: (3, 128)
target = np.random.randn(3, context_len).astype(np.float32)

# 1 past-only covariate channel across past context: (1, 128)
past_only_cov = np.random.randn(1, context_len).astype(np.float32)

# 2 past-and-future covariate channels across context + horizon: (2, 152)
past_future_cov = np.random.randn(2, context_len + horizon).astype(np.float32)

# Generate joint forecast across all 3 target variates
outputs = list(
    forecaster.predict_batch(
        contexts=[target],
        horizon=horizon,
        past_only_covariates=[past_only_cov],
        past_future_covariates=[past_future_cov],
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)

print("Multivariate forecast shape:", outputs[0].forecast.shape)   # (3, 24)
print("Multivariate quantiles shape:", outputs[0].quantiles.shape) # (3, 24, 9)
```
