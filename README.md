# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   All checkpoints:
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/).
*   TimesFM in Google 1P Products:
    *   [BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model): Enterprise level SQL queries for scalability and reliability.
    *   [Google Sheets](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html): For your daily spreadsheet. 
    *   [Vertex Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/timesfm): Dockerized endpoint for agentic calling.

This open version is not an officially supported Google product.

**Latest Model Version:** TimesFM 2.5

**Archived Model Versions:**

-   1.0 and 2.0: relevant code archived in the sub directory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

## Update - Apr. 9, 2026

Added fine-tuning example using HuggingFace Transformers + PEFT (LoRA) — see
[`timesfm-forecasting/examples/finetuning/`](timesfm-forecasting/examples/finetuning/).
Also added unit tests (`tests/`) and incorporated several community fixes.

Shoutout to [@kashif](https://github.com/kashif) and [@darkpowerxo](https://github.com/darkpowerxo). 

## Update - Mar. 19, 2026

Huge shoutout to [@borealBytes](https://github.com/borealBytes) for adding the support for [AGENTS](https://github.com/google-research/timesfm/blob/master/AGENTS.md)! TimesFM [SKILL.md](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting) is out.

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

Since the Sept. 2025 launch, the following improvements have been completed:

1.  ✅ Flax version of the model for faster inference.
2.  ✅ Covariate support via XReg (see Oct. 2025 update).
3.  ✅ Documentation, examples, and agent skill (see `timesfm-forecasting/`).
4.  ✅ Fine-tuning example with LoRA via HuggingFace Transformers + PEFT (see `timesfm-forecasting/examples/finetuning/`).
5.  ✅ Unit tests for core layers, configs, and utilities (see `tests/`).

### Install

TimesFM separates the base package from the heavy ML backends so you only
download what your hardware can use.

**Three optional backends:**

| Backend | What it gives you | Size |
|---|---|---|
| `torch` | PyTorch inference - the primary backend | ~2.5 GB (CUDA) / ~800 MB (CPU) |
| `flax` | JAX/Flax inference - faster on TPU, alternative on GPU | ~300 MB + JAX |
| `xreg` | Covariate / XReg support (requires JAX) | small |

You need at least one of `torch` or `flax` to run the model.
`xreg` is only needed if you use in-context covariates.

---

#### Installing from PyPI

**Step 1 - install the base package:**

```shell
pip install timesfm
```

**Step 2 - let the hardware detector install the right backend:**

`python -m timesfm` detects your GPU, CUDA version, and platform, then installs
the correct extras automatically - no copy-pasting required.

```shell
python -m timesfm --install          # detects hardware, installs torch (prompts once)
python -m timesfm --install --yes    # same, skips the confirmation prompt
```

Running without `--install` is always safe - it only prints what *would* be
installed and exits without changing anything:

```
timesfm hardware detection
========================================
OS       : win32 / AMD64
Python   : 3.12.13
Torch    : NVIDIA GeForce RTX 5070 (12.0 GB VRAM)
JAX      : jax not installed
Backend  : cuda

Recommended install commands:
----------------------------------------
  # torch  (NVIDIA CUDA):
  pip install timesfm[torch] --index-url https://download.pytorch.org/whl/cu128

  # flax   (NVIDIA CUDA 12):
  pip install timesfm[flax-cuda]

  # xreg   (NVIDIA CUDA 12):
  pip install timesfm[xreg-cuda]
```

To install all three backends at once:

```shell
python -m timesfm --install --backend all --yes
```

To install a specific backend only:

```shell
python -m timesfm --install --backend flax --yes
python -m timesfm --install --backend xreg --yes
```

**Hardware coverage:**

| Hardware | Detected by | torch | flax | xreg |
|---|---|---|---|---|
| NVIDIA GPU (CUDA 12) | `nvidia-smi` or torch query | `[torch]` + cu128 index | `[flax-cuda]` | `[xreg-cuda]` |
| NVIDIA GPU (CUDA 11) | torch query | `[torch]` + cu118 index | `[flax-cuda]` | `[xreg-cuda]` |
| Apple Silicon (MPS) | `sys.platform` + `arm64` | `[torch]` (MPS built-in) | `[flax-metal]` | `[xreg-cpu]` |
| Google Cloud TPU | `TPU_NAME` env var | - | `[flax-tpu]` | `[xreg-cpu]` |
| AMD GPU (ROCm) | `rocm-smi` on PATH | `[torch]` + rocm6.2 index | `[flax-cpu]`* | `[xreg-cpu]` |
| CPU only | fallback | `[torch]` | `[flax-cpu]` | `[xreg-cpu]` |

\* JAX ROCm support is experimental. The installer uses `flax-cpu` as a safe
default. For ROCm JAX, follow the
[JAX ROCm instructions](https://jax.readthedocs.io/en/latest/installation.html)
manually.

> **Note for NVIDIA users:** PyTorch CUDA wheels are ~2.5 GB. `uv` caches them
> in `~/.cache/uv` so re-installs across new envs or CI runs are instant after
> the first download.

If you prefer to pick the extras yourself, skip to [Manual install](#manual-install-from-pypi).

---

#### Local development install from source

This path uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment
and dev tools (pytest, ruff, mypy).

```shell
git clone https://github.com/google-research/timesfm.git
cd timesfm
uv sync --all-groups          # creates .venv and installs dev tools
```

Install the ML backend. Because `uv sync` creates an isolated venv, always prefix
with `uv run` so the command runs inside it:

```shell
uv run python -m timesfm --install --yes          # torch for your hardware
uv run python -m timesfm --install --backend all --yes  # torch + flax + xreg
```

Run tests to verify the setup:

```shell
uv run pytest
```

---

#### Manual install from PyPI

If you prefer to control the extras yourself:

**PyTorch:**
```shell
pip install timesfm[torch]                                                        # CPU
pip install timesfm[torch] --index-url https://download.pytorch.org/whl/cu128   # CUDA 12
pip install timesfm[torch] --index-url https://download.pytorch.org/whl/cu118   # CUDA 11
```

**Flax / JAX:**
```shell
pip install timesfm[flax-cpu]     # CPU only
pip install timesfm[flax-cuda]    # NVIDIA GPU (CUDA 12)
pip install timesfm[flax-tpu]     # Google TPU
pip install timesfm[flax-metal]   # Apple Silicon (experimental)
pip install timesfm[flax-auto]    # auto-selects Metal on Apple Silicon, CPU elsewhere
```

**XReg covariates:**
```shell
pip install timesfm[xreg-cpu]     # CPU only
pip install timesfm[xreg-cuda]    # NVIDIA GPU (CUDA 12)
```

### Code Example

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],  # Two dummy inputs
)
point_forecast.shape  # (2, 12)
quantile_forecast.shape  # (2, 12, 10): mean, then 10th to 90th quantiles.
```
