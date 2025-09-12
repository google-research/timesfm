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

"""Forked from https://github.com/Nixtla/nixtla/blob/main/experiments/amazon-chronos/src/utils.py."""

from functools import partial
from itertools import repeat
import multiprocessing
import os
from pathlib import Path
from typing import List

from gluonts.dataset import Dataset
from gluonts.dataset.repository.datasets import (
    dataset_names as gluonts_datasets,
    get_dataset,
)
from gluonts.time_feature.seasonality import get_seasonality
import numpy as np
import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mase, smape


def parallel_transform(inp):
  ts, last_n = inp[0], inp[1]
  return ExperimentHandler._transform_gluonts_instance_to_df(ts, last_n=last_n)


def quantile_loss(
    df: pd.DataFrame,
    models: list,
    q: float = 0.5,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
  delta_y = df[models].sub(df[target_col], axis=0)
  res = (
      np.maximum(q * delta_y, (q - 1) * delta_y)
      .groupby(df[id_col], observed=True)
      .mean()
  )
  res.index.name = id_col
  res = res.reset_index()
  return res


class ExperimentHandler:

  def __init__(
      self,
      dataset: str,
      quantiles: List[float] = list(np.arange(1, 10) / 10.0),
      results_dir: str = "./results",
      models_dir: str = "./models",
  ):
    if dataset not in gluonts_datasets:
      raise Exception(
          f"dataset {dataset} not found in gluonts "
          f"available datasets: {', '.join(gluonts_datasets)}"
      )
    self.dataset = dataset
    self.quantiles = quantiles
    self.level = self._transform_quantiles_to_levels(quantiles)
    self.results_dir = results_dir
    self.models_dir = models_dir
    # defining datasets
    self._maybe_download_m3_or_m5_file(self.dataset)
    gluonts_dataset = get_dataset(self.dataset)
    self.horizon = gluonts_dataset.metadata.prediction_length
    if self.horizon is None:
      raise Exception(
          f"horizon not found for dataset {self.dataset} "
          "experiment cannot be run"
      )
    self.freq = gluonts_dataset.metadata.freq
    # get_seasonality() returns 1 for freq='D', override this to 7. This significantly improves the accuracy of
    # statistical models on datasets like m5/nn5_daily. The models like AutoARIMA/AutoETS can still set
    # seasonality=1 internally on datasets like weather by choosing non-seasonal models during model selection.
    if self.freq == "D":
      self.seasonality = 7
    else:
      self.seasonality = get_seasonality(self.freq)
    self.gluonts_train_dataset = gluonts_dataset.train
    self.gluonts_test_dataset = gluonts_dataset.test
    self._create_dir_if_not_exists(self.results_dir)
    try:
      multiprocessing.set_start_method("spawn")
    except RuntimeError:
      print("Multiprocessing context has already been set.")

  @staticmethod
  def _maybe_download_m3_or_m5_file(dataset: str):
    if dataset[:2] == "m3":
      m3_file = Path.home() / ".gluonts" / "datasets" / "M3C.xls"
      if not m3_file.exists():
        from datasetsforecast.m3 import M3
        from datasetsforecast.utils import download_file

        download_file(m3_file.parent, M3.source_url)
    elif dataset == "m5":
      m5_raw_dir = Path.home() / ".gluonts" / "m5"
      if not m5_raw_dir.exists():
        import zipfile
        from datasetsforecast.m5 import M5
        from datasetsforecast.utils import download_file

        download_file(m5_raw_dir, M5.source_url)
        with zipfile.ZipFile(m5_raw_dir / "m5.zip", "r") as zip_ref:
          zip_ref.extractall(m5_raw_dir)

  @staticmethod
  def _transform_quantiles_to_levels(quantiles: List[float]) -> List[int]:
    level = [
        int(100 - 200 * q) for q in quantiles if q < 0.5
    ]  # in this case mean=mediain
    level = sorted(list(set(level)))
    return level

  @staticmethod
  def _create_dir_if_not_exists(directory: str):
    Path(directory).mkdir(parents=True, exist_ok=True)

  @staticmethod
  def _transform_gluonts_instance_to_df(
      ts: dict,
      last_n: int | None = None,
  ) -> pd.DataFrame:
    start_period = ts["start"]
    start_ds, freq = start_period.to_timestamp(), start_period.freq
    target = ts["target"]
    ds = pd.date_range(start=start_ds, freq=freq, periods=len(target))
    if last_n is not None:
      target = target[-last_n:]
      ds = ds[-last_n:]
    ts_df = pd.DataFrame({"unique_id": ts["item_id"], "ds": ds, "y": target})
    return ts_df

  @staticmethod
  def _transform_gluonts_dataset_to_df(
      gluonts_dataset: Dataset,
      last_n: int | None = None,
  ) -> pd.DataFrame:
    with multiprocessing.Pool(os.cpu_count()) as pool:  # Create a process pool
      results = pool.map(
          parallel_transform, zip(gluonts_dataset, repeat(last_n))
      )
    df = pd.concat(results)
    df = df.reset_index(drop=True)
    return df

  @property
  def train_df(self) -> pd.DataFrame:
    train_df = self._transform_gluonts_dataset_to_df(self.gluonts_train_dataset)
    return train_df

  @property
  def test_df(self) -> pd.DataFrame:
    test_df = self._transform_gluonts_dataset_to_df(
        self.gluonts_test_dataset,
        last_n=self.horizon,
    )
    # Make sure that only the first backtest window is used for evaluation on `traffic` / `exchange_rate` datasets
    return test_df.groupby("unique_id", sort=False).head(self.horizon)

  def save_dataframe(self, df: pd.DataFrame, file_name: str):
    df.to_csv(f"{self.results_dir}/{file_name}", index=False)

  def save_results(
      self, fcst_df: pd.DataFrame, total_time: float, model_name: str
  ):
    self.save_dataframe(
        fcst_df,
        f"{model_name}-{self.dataset}-fcst.csv",
    )
    time_df = pd.DataFrame({"time": [total_time], "model": model_name})
    self.save_dataframe(
        time_df,
        f"{model_name}-{self.dataset}-time.csv",
    )

  def fcst_from_level_to_quantiles(
      self,
      fcst_df: pd.DataFrame,
      model_name: str,
  ) -> pd.DataFrame:
    fcst_df = fcst_df.copy()
    cols = ["unique_id", "ds", model_name]
    for q in self.quantiles:
      if q == 0.5:
        col = f"{model_name}"
      else:
        lv = int(100 - 200 * q)
        hi_or_lo = "lo" if lv > 0 else "hi"
        lv = abs(lv)
        col = f"{model_name}-{hi_or_lo}-{lv}"
      q_col = f"{model_name}-q-{q}"
      fcst_df[q_col] = fcst_df[col].values
      cols.append(q_col)
    return fcst_df[cols]

  def evaluate_models(self, models: List[str]) -> pd.DataFrame:
    fcsts_df = []
    times_df = []
    for model in models:
      fcst_method_df = pd.read_csv(
          f"{self.results_dir}/{model}-{self.dataset}-fcst.csv"
      ).set_index(["unique_id", "ds"])
      fcsts_df.append(fcst_method_df)
      time_method_df = pd.read_csv(
          f"{self.results_dir}/{model}-{self.dataset}-time.csv"
      )
      times_df.append(time_method_df)
    fcsts_df = pd.concat(fcsts_df, axis=1).reset_index()
    fcsts_df["ds"] = pd.to_datetime(fcsts_df["ds"])
    times_df = pd.concat(times_df)
    return self.evaluate_from_predictions(
        models=models, fcsts_df=fcsts_df, times_df=times_df
    )

  def evaluate_from_predictions(
      self, models: List[str], fcsts_df: pd.DataFrame, times_df: pd.DataFrame
  ) -> pd.DataFrame:
    test_df = self.test_df
    train_df = self.train_df
    test_df = test_df.merge(fcsts_df, how="left")
    assert test_df.isna().sum().sum() == 0, "merge contains nas"
    # point evaluation
    point_fcsts_cols = ["unique_id", "ds", "y"] + models
    test_df["unique_id"] = test_df["unique_id"].astype(str)
    train_df["unique_id"] = train_df["unique_id"].astype(str)
    mase_seas = partial(mase, seasonality=self.seasonality)
    eval_df = evaluate(
        test_df[point_fcsts_cols],
        train_df=train_df,
        metrics=[smape, mase_seas, mae],
    )
    # probabilistic evaluation
    eval_prob_df = []
    for q in self.quantiles:
      prob_cols = [f"{model}-q-{q}" for model in models]
      eval_q_df = quantile_loss(test_df, models=prob_cols, q=q)
      eval_q_df[prob_cols] = eval_q_df[prob_cols] * self.horizon
      eval_q_df = eval_q_df.rename(columns=dict(zip(prob_cols, models)))
      eval_q_df["metric"] = f"quantile-loss-{q}"
      eval_prob_df.append(eval_q_df)
    eval_prob_df = pd.concat(eval_prob_df)
    eval_prob_df = eval_prob_df.groupby("metric").sum().reset_index()
    total_y = test_df["y"].sum()
    eval_prob_df[models] = eval_prob_df[models] / total_y
    eval_prob_df["metric"] = "scaled_crps"
    eval_df = pd.concat([eval_df, eval_prob_df]).reset_index(drop=True)
    eval_df = eval_df.groupby("metric").mean(numeric_only=True).reset_index()
    eval_df = eval_df.melt(
        id_vars="metric", value_name="value", var_name="model"
    )
    times_df.insert(0, "metric", "time")
    times_df = times_df.rename(columns={"time": "value"})
    eval_df = pd.concat([eval_df, times_df])
    eval_df.insert(0, "dataset", self.dataset)
    eval_df = eval_df.sort_values(["dataset", "metric", "model"])
    eval_df = eval_df.reset_index(drop=True)
    return eval_df


if __name__ == "__main__":
  multiprocessing.set_start_method("spawn")
