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

"""forecast_csv.py must compile the model for the requested --horizon."""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import timesfm

SCRIPT = Path(__file__).resolve().parents[1] / "timesfm-forecasting/scripts/forecast_csv.py"
spec = importlib.util.spec_from_file_location("forecast_csv", SCRIPT)
forecast_csv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forecast_csv)


class _CompiledModel:
  """Stands in for TimesFM 2.5; like the real one, rejects horizon > max_horizon."""

  def compile(self, forecast_config):
    self.forecast_config = forecast_config

  def forecast(self, horizon, inputs):
    if horizon > self.forecast_config.max_horizon:
      raise ValueError(
        "Horizon must be less than the max horizon."
        f" {horizon} > {self.forecast_config.max_horizon}."
      )
    return np.zeros((len(inputs), horizon)), np.zeros((len(inputs), horizon, 10))


def test_horizon_beyond_default_max_horizon_is_forecast(tmp_path):
  csv_path = tmp_path / "daily.csv"
  pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=30),
    "sales": np.arange(30, dtype=float),
  }).to_csv(csv_path, index=False)
  output_path = tmp_path / "out.csv"
  argv = [
    "forecast_csv.py", str(csv_path), "--horizon", "365", "--date-col", "date",
    "--output", str(output_path), "--skip-check",
  ]

  with mock.patch.object(sys, "argv", argv), mock.patch.object(
    timesfm.TimesFM_2p5_200M_torch, "from_pretrained", return_value=_CompiledModel()
  ):
    forecast_csv.main()

  exported = pd.read_csv(output_path)
  assert len(exported) == 365
  assert exported["step"].max() == 365
