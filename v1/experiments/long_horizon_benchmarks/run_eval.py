# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Eval pipeline."""

import json
import os
import sys
import time
from absl import flags
import chronos
import numpy as np
import pandas as pd
import timesfm
from timesfm import data_loader
import torch
import tqdm

FLAGS = flags.FLAGS

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 64,
                                   "Batch size for the randomly sampled batch")
_DATASET = flags.DEFINE_string("dataset", "etth1", "The name of the dataset.")

_MODEL_PATH = flags.DEFINE_string("model_path", "google/timesfm-2.0-500m-jax",
                                  "The name of the model.")
_DATETIME_COL = flags.DEFINE_string("datetime_col", "date",
                                    "Column having datetime.")
_NUM_COV_COLS = flags.DEFINE_list("num_cov_cols", None,
                                  "Column having numerical features.")
_CAT_COV_COLS = flags.DEFINE_list("cat_cov_cols", None,
                                  "Column having categorical features.")
_TS_COLS = flags.DEFINE_list("ts_cols", None, "Columns of time-series features")
_NORMALIZE = flags.DEFINE_bool("normalize", True,
                               "normalize data for eval or not")
_CONTEXT_LEN = flags.DEFINE_integer("context_len", 2048,
                                    "Length of the context window")
_PRED_LEN = flags.DEFINE_integer("pred_len", 96, "prediction length.")
_BACKEND = flags.DEFINE_string("backend", "gpu", "backend to use")
_RESULTS_DIR = flags.DEFINE_string("results_dir", "./results/long_horizon",
                                   "results directory")

DATA_DICT = {
    "ettm2": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "./datasets/ETT-small/ETTm2.csv",
        "freq": "15min",
    },
    "ettm1": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "./datasets/ETT-small/ETTm1.csv",
        "freq": "15min",
    },
    "etth2": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "./datasets/ETT-small/ETTh2.csv",
        "freq": "H",
    },
    "etth1": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "./datasets/ETT-small/ETTh1.csv",
        "freq": "H",
    },
    "elec": {
        "boundaries": [18413, 21044, 26304],
        "data_path": "./datasets/electricity/electricity.csv",
        "freq": "H",
    },
    "traffic": {
        "boundaries": [12280, 14036, 17544],
        "data_path": "./datasets/traffic/traffic.csv",
        "freq": "H",
    },
    "weather": {
        "boundaries": [36887, 42157, 52696],
        "data_path": "./datasets/weather/weather.csv",
        "freq": "10min",
    },
}

QUANTILES = list(np.arange(1, 10) / 10.0)
EPS = 1e-7


def get_forecasts(model_path, model, past, freq, pred_len):
  """Get forecasts."""
  if model_path.startswith("amazon"):
    out = model.predict(
        torch.tensor(past),
        prediction_length=pred_len,
        limit_prediction_length=False,
    )
    out = out.numpy()
    out = np.median(out, axis=1)
  else:
    lfreq = [freq] * past.shape[0]
    _, out = model.forecast(list(past), lfreq)
    out = out[:, :, 5]
  return out


def _mse(y_pred, y_true):
  """mse loss."""
  return np.square(y_pred - y_true)


def _mae(y_pred, y_true):
  """mae loss."""
  return np.abs(y_pred - y_true)


def _smape(y_pred, y_true):
  """_smape loss."""
  abs_diff = np.abs(y_pred - y_true)
  abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2
  abs_val = np.where(abs_val > EPS, abs_val, 1.0)
  abs_diff = np.where(abs_val > EPS, abs_diff, 0.0)
  return abs_diff / abs_val


def eval():
  """Eval pipeline."""
  dataset = _DATASET.value
  data_path = DATA_DICT[dataset]["data_path"]
  freq = DATA_DICT[dataset]["freq"]
  int_freq = timesfm.freq_map(freq)
  boundaries = DATA_DICT[dataset]["boundaries"]

  data_df = pd.read_csv(open(data_path, "r"))

  if _TS_COLS.value is not None:
    ts_cols = DATA_DICT[dataset]["ts_cols"]
    num_cov_cols = DATA_DICT[dataset]["num_cov_cols"]
    cat_cov_cols = DATA_DICT[dataset]["cat_cov_cols"]
  else:
    ts_cols = [col for col in data_df.columns if col != _DATETIME_COL.value]
    num_cov_cols = None
    cat_cov_cols = None
  batch_size = min(_BATCH_SIZE.value, len(ts_cols))
  dtl = data_loader.TimeSeriesdata(
      data_path=data_path,
      datetime_col=_DATETIME_COL.value,
      num_cov_cols=num_cov_cols,
      cat_cov_cols=cat_cov_cols,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundaries[0]],
      val_range=[boundaries[0], boundaries[1]],
      test_range=[boundaries[1], boundaries[2]],
      hist_len=_CONTEXT_LEN.value,
      pred_len=_PRED_LEN.value,
      batch_size=batch_size,
      freq=freq,
      normalize=_NORMALIZE.value,
      epoch_len=None,
      holiday=False,
      permute=False,
  )
  eval_itr = dtl.tf_dataset(mode="test",
                            shift=_PRED_LEN.value).as_numpy_iterator()
  model_path = _MODEL_PATH.value
  if model_path.startswith("amazon"):
    model = chronos.ChronosPipeline.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
  else:
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=128,
            num_layers=50,
            context_len=_CONTEXT_LEN.value,
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path),
    )
  smape_run_losses = []
  mse_run_losses = []
  mae_run_losses = []

  num_elements = 0
  abs_sum = 0
  start_time = time.time()

  for batch in tqdm.tqdm(eval_itr):
    past = batch[0]
    actuals = batch[3]
    forecasts = get_forecasts(model_path, model, past, int_freq,
                              _PRED_LEN.value)
    forecasts = forecasts[:, 0:actuals.shape[1]]
    mae_run_losses.append(_mae(forecasts, actuals).sum())
    mse_run_losses.append(_mse(forecasts, actuals).sum())
    smape_run_losses.append(_smape(forecasts, actuals).sum())
    num_elements += actuals.shape[0] * actuals.shape[1]
    abs_sum += np.abs(actuals).sum()

  mse_val = np.sum(mse_run_losses) / num_elements

  result_dict = {
      "mse": mse_val,
      "smape": np.sum(smape_run_losses) / num_elements,
      "mae": np.sum(mae_run_losses) / num_elements,
      "wape": np.sum(mae_run_losses) / abs_sum,
      "nrmse": np.sqrt(mse_val) / (abs_sum / num_elements),
      "num_elements": num_elements,
      "abs_sum": abs_sum,
      "total_time": time.time() - start_time,
      "model_path": model_path,
      "dataset": dataset,
      "freq": freq,
      "pred_len": _PRED_LEN.value,
      "context_len": _CONTEXT_LEN.value,
  }
  run_id = np.random.randint(10000)
  save_path = os.path.join(_RESULTS_DIR.value, str(run_id))
  print(f"Saving results to {save_path}", flush=True)
  os.makedirs(save_path, exist_ok=True)
  with open(os.path.join(save_path, "results.json"), "w") as f:
    json.dump(result_dict, f)
  print(result_dict, flush=True)


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  FLAGS(sys.argv)
  eval()
