# Copyright 2026 Google LLC
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

"""forecast_csv.py must extend dates without pandas frequency warnings."""

import importlib.util
import warnings
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "timesfm-forecasting/scripts/forecast_csv.py"
spec = importlib.util.spec_from_file_location("forecast_csv", SCRIPT)
forecast_csv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forecast_csv)


def _results(horizon):
  keys = ["forecast", "lower_90", "lower_80", "median", "upper_80", "upper_90"]
  return {"y": {k: [0.0] * horizon for k in keys}}


def test_write_csv_output_extends_monthly_dates_without_warnings(tmp_path):
  df = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
      "y": range(6),
    }
  )
  out = tmp_path / "forecast.csv"

  with warnings.catch_warnings():
    warnings.simplefilter("error")
    forecast_csv.write_csv_output(_results(3), str(out), df, "date", 3)

  written = pd.read_csv(out, parse_dates=["date"])
  assert written["date"].tolist() == list(
    pd.to_datetime(["2024-07-01", "2024-08-01", "2024-09-01"])
  )


def test_infer_freq_returns_none_for_irregular_dates():
  dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]))
  assert forecast_csv.infer_freq(dates) is None
