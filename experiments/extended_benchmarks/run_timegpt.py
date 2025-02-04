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

"""Evaluation script for timegpt."""

import os
import sys
import time

from absl import flags
import numpy as np
import pandas as pd

from ..baselines.timegpt_pipeline import run_timegpt
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

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "timegpt-1-long-horizon",
    "Path to model, can also be set to timegpt-1",
)
_SAVE_DIR = flags.DEFINE_string("save_dir", "./results", "Save directory")


QUANTILES = list(np.arange(1, 10) / 10.0)


def main():
  results_list = []
  run_id = np.random.randint(100000)
  model_name = _MODEL_NAME.value
  for dataset in dataset_names:
    print(f"Evaluating model {model_name} on dataset {dataset}", flush=True)
    exp = ExperimentHandler(dataset, quantiles=QUANTILES)
    train_df = exp.train_df
    horizon = exp.horizon
    seasonality = exp.seasonality
    freq = exp.freq
    level = exp.level
    fcsts_df, total_time, model_name = run_timegpt(
        train_df=train_df,
        horizon=exp.horizon,
        model=model_name,
        seasonality=seasonality,
        freq=freq,
        dataset=dataset,
        level=level,
    )
    time_df = pd.DataFrame({"time": [total_time], "model": model_name})
    fcsts_df = exp.fcst_from_level_to_quantiles(fcsts_df, model_name)
    results = exp.evaluate_from_predictions(
        models=[model_name], fcsts_df=fcsts_df, times_df=time_df
    )
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
