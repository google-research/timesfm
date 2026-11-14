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

"""Interval columns written by forecast_csv.py must match their coverage."""

import importlib.util
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_base import TimesFM_2p5

SCRIPT = (
  Path(__file__).resolve().parents[1] / "timesfm-forecasting/scripts/forecast_csv.py"
)
spec = importlib.util.spec_from_file_location("forecast_csv", SCRIPT)
forecast_csv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forecast_csv)

HORIZON = 3


def quantile_level_model():
  """A weightless TimesFM 2.5 whose quantile forecast is the quantile level.

  Column 0 is the mean and columns 1..9 are the 0.1..0.9 quantiles, so each
  exported bound reads back as the quantile level it was taken from.
  """
  model = TimesFM_2p5()
  model.forecast_config = ForecastConfig(max_context=8, max_horizon=HORIZON)
  model.global_batch_size = 1
  levels = np.array([0.5] + [k / 10 for k in range(1, 10)])
  model.compiled_decode = mock.Mock(
    return_value=(np.full((1, HORIZON), 0.5), np.tile(levels, (1, HORIZON, 1)))
  )
  return model


def interval_names(columns):
  lowers = {c[len("lower_") :] for c in columns if c.startswith("lower_")}
  uppers = {c[len("upper_") :] for c in columns if c.startswith("upper_")}
  assert lowers and lowers == uppers
  return sorted(lowers)


def assert_nominal_coverage(table):
  # A central interval between the q and 1 - q quantiles covers 1 - 2q.
  for name in interval_names(table):
    coverage = int(name) / 100
    lower = np.asarray(table[f"lower_{name}"], dtype=float)
    upper = np.asarray(table[f"upper_{name}"], dtype=float)
    np.testing.assert_allclose(lower, (1 - coverage) / 2, err_msg=name)
    np.testing.assert_allclose(upper, (1 + coverage) / 2, err_msg=name)


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_interval_names_match_their_coverage(tmp_path, fmt):
  frame = pd.DataFrame({"sales": np.arange(8, dtype=np.float32)})
  results = forecast_csv.forecast_series(
    quantile_level_model(), frame, ["sales"], HORIZON
  )

  output_path = tmp_path / f"forecast.{fmt}"
  if fmt == "csv":
    forecast_csv.write_csv_output(results, str(output_path), frame, None, HORIZON)
    table = pd.read_csv(output_path)
  else:
    forecast_csv.write_json_output(results, str(output_path))
    table = json.loads(output_path.read_text())["sales"]

  assert_nominal_coverage(table)
  np.testing.assert_allclose(table["median"], 0.5)
