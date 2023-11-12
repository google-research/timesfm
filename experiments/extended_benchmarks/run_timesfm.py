# Copyright 2024 Google LLC
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
"""Evaluation script for timesfm."""

import os
import sys
import time

from absl import flags
import numpy as np
import pandas as pd
import timesfm

from .utils import ExperimentHandler

dataset_names = [
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    "m3_monthly",
    "m3_other",
    "m3_quarterly",
    "m3_yearly",
    "m4_quarterly",
    "m4_yearly",
    "tourism_monthly",
    "tourism_quarterly",
    "tourism_yearly",
    "nn5_daily_without_missing",
    "m5",
    "nn5_weekly",
    "traffic",
    "weather",
    "australian_electricity_demand",
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "ercot",
    "ett_small_15min",
    "ett_small_1h",
    "exchange_rate",
    "fred_md",
    "hospital",
]


context_dict_v2 = {}

context_dict_v1 = {
    "cif_2016": 32,
    "tourism_yearly": 64,
    "covid_deaths": 64,
    "tourism_quarterly": 64,
    "tourism_monthly": 64,
    "m1_monthly": 64,
    "m1_quarterly": 64,
    "m1_yearly": 64,
    "m3_monthly": 64,
    "m3_other": 64,
    "m3_quarterly": 64,
    "m3_yearly": 64,
    "m4_quarterly": 64,
    "m4_yearly": 64,
}

_MODEL_PATH = flags.DEFINE_string("model_path", "google/timesfm-2.0-500m-jax",
                                  "Path to model")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 64, "Batch size")
_HORIZON = flags.DEFINE_integer("horizon", 128, "Horizon")
_BACKEND = flags.DEFINE_string("backend", "gpu", "Backend")
_NUM_JOBS = flags.DEFINE_integer("num_jobs", 1, "Number of jobs")
_SAVE_DIR = flags.DEFINE_string("save_dir", "./results", "Save directory")

QUANTILES = list(np.arange(1, 10) / 10.0)


def main():
  results_list = []
  model_path = _MODEL_PATH.value
  num_layers = 20
  max_context_len = 512
  use_positional_embedding = True
  context_dict = context_dict_v1
  if "2.0" in model_path:
    num_layers = 50
    use_positional_embedding = False
    max_context_len = 2048
    context_dict = context_dict_v2

  tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=num_layers,
          context_len=max_context_len,
          use_positional_embedding=use_positional_embedding,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path),
  )
  run_id = np.random.randint(100000)
  model_name = "timesfm"
  for dataset in dataset_names:
    print(f"Evaluating model {model_name} on dataset {dataset}", flush=True)
    exp = ExperimentHandler(dataset, quantiles=QUANTILES)

    if dataset in context_dict:
      context_len = context_dict[dataset]
    else:
      context_len = max_context_len

    train_df = exp.train_df
    freq = exp.freq
    init_time = time.time()
    fcsts_df = tfm.forecast_on_df(
        inputs=train_df,
        freq=freq,
        value_name="y",
        model_name=model_name,
        forecast_context_len=context_len,
        num_jobs=_NUM_JOBS.value,
        normalize=True,
    )
    total_time = time.time() - init_time
    time_df = pd.DataFrame({"time": [total_time], "model": model_name})
    results = exp.evaluate_from_predictions(models=[model_name],
                                            fcsts_df=fcsts_df,
                                            times_df=time_df)
    print(results, flush=True)
    results_list.append(results)
    results_full = pd.concat(results_list)
    save_path = os.path.join(_SAVE_DIR.value, str(run_id))
    print(f"Saving results to {save_path}", flush=True)
    os.makedirs(save_path, exist_ok=True)
    results_full.to_csv(f"{save_path}/results.csv")


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  FLAGS(sys.argv)
  main()
