# TimesFM

TimesFM  (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google
Research for time-series forecasting.

* Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688), to appear in ICML 2024.
* [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
* [Hugging Face checkpoint repo](https://huggingface.co/google/timesfm-1.0-200m)

This repo contains the code to load public TimesFM checkpoints and run model
inference. Please visit our 
[Hugging Face checkpoint repo](https://huggingface.co/google/timesfm-1.0-200m)
to download model checkpoints.

This is not an officially supported Google product.

We recommend at least 16GB RAM to load TimesFM dependencies.

## Update - July 15, 2024

- To install TimesFM, you can now simply do: `pip install timesfm`.
- Launched [finetuning support](https://github.com/google-research/timesfm/blob/master/notebooks/finetuning.ipynb) that lets you finetune the weights of the pretrained TimesFM model on your own data.
- Launched [~zero-shot covariate support](https://github.com/google-research/timesfm/blob/master/notebooks/covariates.ipynb) with external regressors. More details [here](https://github.com/google-research/timesfm?tab=readme-ov-file#covariates-support).

## Checkpoint timesfm-1.0-200m

timesfm-1.0-200m is the first open model checkpoint:

- It performs univariate time series forecasting for context lengths up to 512 timepoints and any horizon lengths, with an optional frequency indicator.
- It focuses on point forecasts, and does not support probabilistic forecasts. We experimentally offer quantile heads but they have not been calibrated after pretraining.
- It requires the context to be contiguous (i.e. no "holes"), and the context and the horizon to be of the same frequency.

## Benchmarks

Please refer to our result tables on the [extended benchmarks](https://github.com/google-research/timesfm/tree/master/experiments/extended_benchmarks) and the [long horizon benchmarks](https://github.com/google-research/timesfm/tree/master/experiments/long_horizon_benchmarks).

Please look into the README files in the respective benchmark directories within `experiments/` for instructions for running TimesFM on the respective benchmarks.

## Installation

### Installation as a package

To install the TimesFM as a package, you can run the following command without cloning this repo:

`pip install timesfm`

### Installation using conda

For calling TimesFM, We have two environment files. Inside `timesfm`, for
GPU installation (assuming CUDA 12 has been setup), you can create a conda
environment `tfm_env` from the base folder through:

```
conda env create --file=environment.yml
```

For a CPU setup please use,

```
conda env create --file=environment_cpu.yml
```
to create the environment instead.

Follow by

```
conda activate tfm_env
pip install -e .
```
to install the package.

**Note**: 

1. Running the provided benchmarks would require additional dependencies.
Please use the environment files under `experiments` instead.

2. The dependency `lingvo` does not support ARM architectures, and the code is not working for machines with Apple silicon. We are aware of this issue and are working on a solution. Stay tuned.


### Local installation using poetry

To from the current repository/local version (like you would have previously done with `pip -e .`), you can run the command

```
pip install poetry # optional
poetry install
```

This will install the environment in the local .venv folder (depends on the configuration) and matches the python command to the poetry environment. If this is not the case, you can use `poetry run python` to use the local environment.

### Notes

1. Running the provided benchmarks would require additional dependencies.
Please use the environment files under `experiments` instead.

2. The dependency `lingvo` does not support ARM architectures, and the code is not working for machines with Apple silicon. We are aware of this issue and are working on a solution. Stay tuned.

#### Building the package and publishing to PyPI

The package can be built using the command `poetry build`.

To build and publish it to PyPI, the command `poetry publish` can be used. This command will require the user to have the necessary permissions to publish to the PyPI repository.

## Usage 

### Initialize the model and load a checkpoint.
Then the base class can be loaded as,

```python
import timesfm

tfm = timesfm.TimesFm(
    context_len=<context>,
    horizon_len=<horizon>,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend=<backend>,
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
```

Note that the four parameters are fixed to load the 200m model

```python
input_patch_len=32,
output_patch_len=128,
num_layers=20,
model_dims=1280,
```

1. The `context_len` here can be set as the max context length **of the model**. **It needs to be a multiplier of `input_patch_len`, i.e. a multiplier of 32.** You can provide a shorter series to the `tfm.forecast()` function and the model will handle it. Currently, the model handles a max context length of 512, which can be increased in later releases. The input time series can have **any context length**. Padding / truncation will be handled by the inference code if needed.

2. The horizon length can be set to anything. We recommend setting it to the largest horizon length you would need in the forecasting tasks for your application. We generally recommend horizon length <= context length but it is not a requirement in the function call.

3. `backend` is one of "cpu", "gpu" or "tpu", case sensitive.

### Perform inference

We provide APIs to forecast from either array inputs or `pandas` dataframe. Both forecast methods expect (1) the input time series contexts, (2) along with their frequencies. Please look at the documentation of the functions `tfm.forecast()` and `tfm.forecast_on_df()` for detailed instructions.

In particular regarding the frequency, TimesFM expects a categorical indicator valued in {0, 1, 2}:

- **0** (default): high frequency, long horizon time series. We recommend using this for time series up to daily granularity.
- **1**: medium frequency time series. We recommend using this for weekly and monthly data.
- **2**: low frequency, short horizon time series. We recommend using this for anything beyond monthly, e.g. quarterly or yearly.

This categorical value should be directly provided with the array inputs. For dataframe inputs, we convert the conventional letter coding of frequencies to our expected categories, that

- **0**: T, MIN, H, D, B, U
- **1**: W, M
- **2**: Q, Y

Notice you do **NOT** have to strictly follow our recommendation here. Although this is our setup during model training and we expect it to offer the best forecast result, you can also view the frequency input as a free parameter and modify it per your specific use case.


Examples:

Array inputs, with the frequencies set to low, medium and high respectively.

```python
import numpy as np
forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
frequency_input = [0, 1, 2]

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)
```

`pandas` dataframe, with the frequency set to "M" monthly.

```python
import pandas as pd

# e.g. input_df is
#       unique_id  ds          y
# 0     T1         1975-12-31  697458.0
# 1     T1         1976-01-31  1187650.0
# 2     T1         1976-02-29  1069690.0
# 3     T1         1976-03-31  1078430.0
# 4     T1         1976-04-30  1059910.0
# ...   ...        ...         ...
# 8175  T99        1986-01-31  602.0
# 8176  T99        1986-02-28  684.0
# 8177  T99        1986-03-31  818.0
# 8178  T99        1986-04-30  836.0
# 8179  T99        1986-05-31  878.0

forecast_df = tfm.forecast_on_df(
    inputs=input_df,
    freq="M",  # monthly
    value_name="y",
    num_jobs=-1,
)
```

## Covariates Support

We now have an external regressors library on top of TimesFM that can support static covariates as well as dynamic covariates available in the future. We have an usage example in [notebooks/covariates.ipynb](https://github.com/google-research/timesfm/blob/master/notebooks/covariates.ipynb).

Let's take a toy example of forecasting sales for a grocery store: 

**Task:** Given the observed the daily sales of this week (7 days), forecast the daily sales of next week (7 days).

```
Product: ice cream
Daily_sales: [30, 30, 4, 5, 7, 8, 10]
Category: food
Base_price: 1.99
Weekday: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
Has_promotion: [Yes, Yes, No, No, No, Yes, Yes, No, No, No, No, No, No, No]
Daily_temperature: [31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1, 32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2]
```

```
Product: sunscreen
Daily_sales: [5, 7, 12, 13, 5, 6, 10]
Category: skin product
Base_price: 29.99
Weekday: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
Has_promotion: [No, No, Yes, Yes, No, No, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes]
Daily_temperature: [31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1, 32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2]
```

In this example, besides the `Daily_sales`, we also have covariates `Category`, `Base_price`, `Weekday`, `Has_promotion`, `Daily_temperature`. Let's introduce some concepts:

**Static covariates** are covariates for each time series. 
- In our example, `Category` is a **static categorical covariate**, 
- `Base_price` is a **static numerical covariates**.

**Dynamic covariates** are covaraites for each time stamps.
- Date / time related features can be usually treated as dynamic covariates.
- In our example, `Weekday` and `Has_promotion` are **dynamic categorical covariates**.
- `Daily_temperate` is a **dynamic numerical covariate**.

**Notice:** Here we make it mandatory that the dynamic covariates need to cover both the forecasting context and horizon. For example, all dynamic covariates in the example have 14 values: the first 7 correspond to the observed 7 days, and the last 7 correspond to the next 7 days.

We can now provide the past data of the two products along with static and dynamic covariates as a batch input to TimesFM and produce forecasts that take into the account the covariates. To learn more, check out the example in [notebooks/covariates.ipynb](https://github.com/google-research/timesfm/blob/master/notebooks/covariates.ipynb).

## Finetuning

We have provided an example of finetuning the model on a new dataset in [notebooks/finetuning.ipynb](https://github.com/google-research/timesfm/blob/master/notebooks/finetuning.ipynb).

## Contribution Style guide

If you would like to submit a PR please make sure that you use our formatting style. We use [yapf](https://github.com/google/yapf) for formatting with the following options,

```
[style]
based_on_style = google
# Add your custom style rules here
indent_width = 2
spaces_before_comment = 2

```

Please run `yapf --in-place --recursive <filename>` on all affected files.
