# TimesFM

TimesFM  (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google
Research for time-series forecasting.

* Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688), to appear in ICML 2024.
* [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
* [Hugging Face release](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)

This repo contains the code to load public TimesFM checkpoints and run model
inference. Please visit our 
[Hugging Face release](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)
to download model checkpoints.

This is not an officially supported Google product.

We recommend at least 32GB RAM to load TimesFM dependencies.

**Need help?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common installation and usage issues.

## Update - Dec. 30, 2024
- We are launching a 500m checkpoint as a part of TimesFM-2.0 release. This new checkpoint can be upto 25% better than v1.0 on leading benchmarks and also has a 4 times longer max. context length.
- Launched [finetuning support](https://github.com/google-research/timesfm/blob/master/notebooks/finetuning.ipynb) that lets you finetune the weights of the pretrained TimesFM model on your own data.
- Launched [~zero-shot covariate support](https://github.com/google-research/timesfm/blob/master/notebooks/covariates.ipynb) with external regressors. More details [here](https://github.com/google-research/timesfm?tab=readme-ov-file#covariates-support).

## Update - Feb. 17, 2024
- We are providing the option for [finetuning using Pytorch](https://github.com/google-research/timesfm/blob/master/notebooks/finetuning_torch.ipynb), which mimics the previously added functionality from [finetuning support](https://github.com/google-research/timesfm/blob/master/notebooks/finetuning.ipynb).
- We are also providing the Multi-GPU finetuining with Pytorch. We currently support DDP multi-gpu finetuning, other variants of multi-gpu training (pipeline parallelism/model parallelism) might be added later. In order to use it, follow the steps in [finetuning example](https://github.com/google-research/timesfm/blob/master/finetuning/finetuning_example.py) .

## Checkpoint timesfm-1.0-200m (-pytorch)

timesfm-1.0-200m is our first open model checkpoint:

- It performs univariate time series forecasting for context lengths up to 512 timepoints and any horizon lengths, with an optional frequency indicator.
- It focuses on point forecasts, and does not support probabilistic forecasts. We experimentally offer quantile heads but they have not been calibrated after pretraining.

## Checkpoint timesfm-2.0-500m (-jax/-pytorch)

timesfm-2.0-500m is our second open model checkpoint:

- It performs univariate time series forecasting for context lengths up to 2048 timepoints and any horizon lengths, with an optional frequency indicator.
- It focuses on point forecasts. We experimentally offer 10 quantile heads but they have not been calibrated after pretraining.
- This new checkpoint can be upto 25% better than v1.0 on leading benchmarks and also has a 4 times longer max. context length.

## Benchmarking

TimesFM 2.0 has been added to [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) which is one of the most comprehensive time-series bechmarks available. It takes the top spot in terms of aggregated MASE and CRPS, where it is 6\% better than the next best model in terms of aggregated MASE.

## Installation

### Local installation using poetry

We will be using `pyenv` and `poetry`. In order to set these things up please follow the instructions [here](https://substack.com/home/post/p-148747960?r=28a5lx&utm_campaign=post&utm_medium=web). Note that the PAX (or JAX) version needs to run on python 3.10.x and the PyTorch version can run on >=3.11.x. Therefore make sure you have two versions of python installed:

```
pyenv install 3.10
pyenv install 3.11
pyenv versions # to list the versions available (lets assume the versions are 3.10.15 and 3.11.10)
```

### For PAX version installation do the following.

```
pyenv local 3.10.15
poetry env use 3.10.15
poetry lock
poetry install -E  pax
```

After than you can run the timesfm under `poetry shell` or do `poetry run python3 ...`.

### For PyTorch version installation do the following.

```
pyenv local 3.11.10
poetry env use 3.11.10
poetry lock
poetry install -E  torch
```

After than you can run the timesfm under `poetry shell` or do `poetry run python3 ...`.

**Additional Note**: 

If you plan to use the **`forecast_with_covariates`** function (which requires external regressors), 
you need to install **JAX** and **jaxlib**. If you installed the base version of TimesFM (`torch`), you must manually install the dependencies for **`forecast_with_covariates`** support:
```
pip install jax jaxlib
```

**Why is this needed?**  
The `forecast_with_covariates` method relies on the `xreg_lib` module, which depends on JAX and jaxlib. If these packages are not installed, 
calling `forecast_with_covariates` will raise an error. However, due to a lazy import mechanism, `xreg_lib` (and hence JAX/jaxlib) is not needed for standard `forecast` calls.

### Notes

1. Running the provided benchmarks would require additional dependencies. Please see the `experiments` folder.

2. The dependency `lingvo` does not support ARM architectures, and the code is not working for machines with Apple silicon. We are aware of this issue and are working on a solution. Stay tuned.

### Install from PyPI (and publish)

On python 3.11 you can install the torch version using:

```pip install timesfm[torch]```

On python 3.10 you can install the pax version using:

```pip install timesfm[pax]```


## Usage 

### Initialize the model and load a checkpoint.
Then the base class can be loaded as,

```python
import timesfm

# Loading the timesfm-2.0 checkpoint:
# For PAX
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=50,
          context_len=2048,

          use_positional_embedding=False,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-jax"),
  )

# For Torch
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
  )

# Loading the timesfm-1.0 checkpoint:
# For PAX
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m"),
  )

# For Torch
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
  )
```

Note some of the parameters are fixed to load the 200m and 500m models

1. The `context_len` in `hparams` here can be set as the max context length **of the model** (a maximum of 2048 for 2.0 models and 512 for 1.0 models). **It needs to be a multiplier of `input_patch_len`, i.e. a multiplier of 32.** You can provide a shorter series to the `tfm.forecast()` function and the model will handle it. The input time series can have **any context length**. Padding / truncation will be handled by the inference code if needed.

2. The horizon length can be set to anything. We recommend setting it to the largest horizon length you would need in the forecasting tasks for your application. We generally recommend horizon length <= context length but it is not a requirement in the function call.

3. `backend` is one of "cpu", "gpu", case sensitive.

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

If you plan to use the **`forecast_with_covariates`** on timesfm `torch` version, you need to install **JAX** and **jaxlib**. 
You must manually install the dependencies for **`forecast_with_covariates`** support:
```
pip install jax jaxlib
```

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
