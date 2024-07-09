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

import os
from time import time
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from gluonts.time_feature.seasonality import get_seasonality as _get_seasonality
from nixtla import NixtlaClient
import pandas as pd
from tqdm import tqdm
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)


def get_seasonality(freq: str) -> int:
  return _get_seasonality(freq, seasonalities={"D": 7})


def maybe_convert_col_to_datetime(
    df: pd.DataFrame, col_name: str
) -> pd.DataFrame:
  if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
    df = df.copy()
    df[col_name] = pd.to_datetime(df[col_name])
  return df


def zero_pad_time_series(df, freq, min_length=36):
  """If time_series length is less than min_length, front pad it with zeros."""
  # 1. Calculate required padding for each unique_id
  value_counts = df["unique_id"].value_counts()
  to_pad = value_counts[value_counts < min_length].index

  # 2. Create a new DataFrame to hold padded data
  padded_data = []

  for unique_id in to_pad:
    # 2a. Filter data for the specific unique_id
    subset = df[df["unique_id"] == unique_id]
    if len(subset) > min_length:
      padded_data.append(subset)
    else:
      # 2b. Determine earliest date and calculate padding dates
      start_date = subset["ds"].min()
      padding_dates = pd.date_range(
          end=start_date,
          periods=min_length - len(subset) + 1,
          freq=freq,  # 'MS' for month start
      )[
          :-1
      ]  # Exclude the start_date itself

      # 2c. Create padding data
      padding_df = pd.DataFrame(
          {"ds": padding_dates, "unique_id": unique_id, "y": 0}  # Zero padding
      )

      # 2d. Combine original and padding data, and append to the list
      padded_data.append(pd.concat([padding_df, subset]).sort_values("ds"))

  # 3. Combine all padded data and original data (unchanged)
  result_df = pd.concat(padded_data + [df[~df["unique_id"].isin(to_pad)]])
  return result_df


class Forecaster:
  """Borrowed from

  https://github.com/Nixtla/nixtla/tree/main/experiments/foundation-time-series-arena/xiuhmolpilli/models.
  """

  def forecast(
      self,
      df: pd.DataFrame,
      h: int,
      freq: str,
  ) -> pd.DataFrame:
    raise NotImplementedError

  def cross_validation(
      self,
      df: pd.DataFrame,
      h: int,
      freq: str,
      n_windows: int = 1,
      step_size: int | None = None,
  ) -> pd.DataFrame:
    df = maybe_convert_col_to_datetime(df, "ds")
    # mlforecast cv code
    results = []
    sort_idxs = maybe_compute_sort_indices(df, "unique_id", "ds")
    if sort_idxs is not None:
      df = take_rows(df, sort_idxs)
    splits = backtest_splits(
        df,
        n_windows=n_windows,
        h=h,
        id_col="unique_id",
        time_col="ds",
        freq=pd.tseries.frequencies.to_offset(freq),
        step_size=h if step_size is None else step_size,
    )
    for _, (cutoffs, train, valid) in tqdm(enumerate(splits)):
      if len(valid.columns) > 3:
        raise NotImplementedError(
            "Cross validation with exogenous variables is not yet supported."
        )
      y_pred = self.forecast(
          df=train,
          h=h,
          freq=freq,
      )
      y_pred = join(y_pred, cutoffs, on="unique_id", how="left")
      result = join(
          valid[["unique_id", "ds", "y"]],
          y_pred,
          on=["unique_id", "ds"],
      )
      if result.shape[0] < valid.shape[0]:
        raise ValueError(
            "Cross validation result produced less results than expected."
            " Please verify that the frequency parameter (freq) matches your"
            " series' and that there aren't any missing periods."
        )
      results.append(result)
    out = vertical_concat(results)
    out = drop_index_if_pandas(out)
    first_out_cols = ["unique_id", "ds", "cutoff", "y"]
    remaining_cols = [c for c in out.columns if c not in first_out_cols]
    fcst_cv_df = out[first_out_cols + remaining_cols]
    return fcst_cv_df


class TimeGPT(Forecaster):
  """Borrowed from

  https://github.com/Nixtla/nixtla/tree/main/experiments/foundation-time-series-arena/xiuhmolpilli/models.
  We modify the class to take care of edge cases.
  """

  def __init__(
      self,
      api_key: str | None = None,
      base_url: Optional[str] = None,
      max_retries: int = 1,
      model: str = "timegpt-1",
      alias: str = "TimeGPT",
  ):
    self.api_key = api_key
    self.base_url = base_url
    self.max_retries = max_retries
    self.model = model
    self.alias = alias

  def _get_client(self) -> NixtlaClient:
    if self.api_key is None:
      api_key = os.environ["NIXTLA_API_KEY"]
    else:
      api_key = self.api_key
    return NixtlaClient(
        api_key=api_key,
        base_url=self.base_url,
        max_retries=self.max_retries,
    )

  def forecast(
      self,
      df: pd.DataFrame,
      h: int,
      freq: str,
      level: List = [90.0],
      chunk_size: Optional[int] = None,
  ) -> pd.DataFrame:
    client = self._get_client()
    fcst_df = None
    if chunk_size is None:
      fcst_df = client.forecast(
          df=df,
          h=h,
          freq=freq,
          level=level,
          model=self.model,
      )
    else:
      all_unique_ids = df["unique_id"].unique()
      all_fcst_df = []
      for i in range(0, len(all_unique_ids), chunk_size):
        chunk_ids = all_unique_ids[i : i + chunk_size]
        chunk_df = df[df["unique_id"].isin(chunk_ids)]
        fct_chunk_df = client.forecast(
            df=chunk_df,
            h=h,
            freq=freq,
            level=level,
        )
        all_fcst_df.append(fct_chunk_df)
      fcst_df = pd.concat(all_fcst_df)
    fcst_df["ds"] = pd.to_datetime(fcst_df["ds"])
    replace_dict = {}
    for col in fcst_df.columns:
      if col.startswith("TimeGPT"):
        replace_dict[col] = col.replace("TimeGPT", self.alias)
    fcst_df = fcst_df.rename(columns=replace_dict)
    return fcst_df


def run_timegpt(
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
    seasonality: int,
    level: List[int],
    dataset: str,
    model: str = "timegpt-1",
) -> Tuple[pd.DataFrame, float, str]:
  os.environ["NIXTLA_ID_AS_COL"] = "true"
  model = TimeGPT(model="timegpt-1", alias=model)
  padded_train_df = zero_pad_time_series(train_df, freq)
  init_time = time()
  # For these datasets the API fails if we do not chunk.
  if dataset in ["m5", "m4_quarterly"]:
    chunk_size = 5000
  else:
    chunk_size = None
  fcsts_df = model.forecast(
      df=padded_train_df,
      h=horizon,
      level=level,
      freq=freq,
      chunk_size=chunk_size,
  )
  total_time = time() - init_time
  # In case levels are not returned we replace the levels with the mean predictions.
  # Note that this does not affect the results table as we only compare on point
  # forecastign metrics.
  for lvl in level:
    if f"{model.alias}-lo-{lvl}" not in fcsts_df.columns:
      fcsts_df[f"{model.alias}-lo-{lvl}"] = fcsts_df[model.alias]
    if f"{model.alias}-hi-{lvl}" not in fcsts_df.columns:
      fcsts_df[f"{model.alias}-hi-{lvl}"] = fcsts_df[model.alias]
  return fcsts_df, total_time, model.alias
