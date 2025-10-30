# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   All checkpoints:
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/).
*   [TimesFM in BigQuery](https://cloud.google.com/bigquery/docs/timesfm-model):
    an official Google product.

This open version is not an officially supported Google product.

**Latest Model Version:** TimesFM 2.5

**Archived Model Versions:**

-   1.0 and 2.0: relevant code archived in the sub directory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

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

Along with the model upgrade we have also upgraded the inference API. This repo
will be under construction over the next few weeks to

1.  add support for an upcoming Flax version of the model (faster inference).
2.  add back covariate support.
3.  populate more docstrings, docs and notebook.

### Install

1.  Clone the repository:
    ```shell
    git clone https://github.com/google-research/timesfm.git
    cd timesfm
    ```

2.  Create a virtual environment and install dependencies using `uv`:
    ```shell
    # Create a virtual environment
    uv venv
    
    # Activate the environment
    source .venv/bin/activate
    
    # Install the package in editable mode with torch
    uv pip install -e .[torch]
    # Or with flax
    uv pip install -e .[flax]
    # Or XReg is needed
    uv pip install -e .[xreg]
    ```

3. [Optional] Install your preferred `torch` / `jax` backend based on your OS and accelerators
(CPU, GPU, TPU or Apple Silicon).:

-   [Install PyTorch](https://pytorch.org/get-started/locally/).
-   [Install Jax](https://docs.jax.dev/en/latest/installation.html#installation)
    for Flax.

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
