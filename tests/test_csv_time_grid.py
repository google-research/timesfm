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

"""Missing observations must retain their original time positions."""

import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_base import TimesFM_2p5

SCRIPT = Path(__file__).resolve().parents[1] / "timesfm-forecasting/scripts/forecast_csv.py"
spec = importlib.util.spec_from_file_location("forecast_csv", SCRIPT)
forecast_csv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forecast_csv)


def preprocessing_model():
  """Run TimesFM 2.5 forecast preprocessing without loading model weights."""
  model = TimesFM_2p5()
  model.forecast_config = ForecastConfig(max_context=5)
  model.global_batch_size = 2
  model.compiled_decode = mock.Mock(
    return_value=(np.zeros((2, 2)), np.zeros((2, 2, 10)))
  )
  model.forecast = mock.Mock(wraps=model.forecast)
  return model


def test_missing_observations_keep_shared_time_grid(tmp_path):
  frame = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=5),
    "sales": [1.0, np.nan, 3.0, 4.0, np.nan],
    "demand": [np.nan, 2.0, 3.0, np.nan, 5.0],
  })
  original = frame.copy(deep=True)
  model = preprocessing_model()

  results = forecast_csv.forecast_series(model, frame, ["sales", "demand"], 2)

  inputs = model.forecast.call_args.kwargs["inputs"]
  for column, values in zip(["sales", "demand"], inputs):
    assert values.dtype == np.float32
    assert values.shape == (len(frame),)
    np.testing.assert_array_equal(values, frame[column].to_numpy(dtype=np.float32))
  _, preprocessed, masks = model.compiled_decode.call_args.args
  np.testing.assert_array_equal(preprocessed[0], [1.0, 2.0, 3.0, 4.0, 4.0])
  np.testing.assert_array_equal(preprocessed[1], [0.0, 2.0, 3.0, 4.0, 5.0])
  np.testing.assert_array_equal(masks[0], [False] * 5)
  np.testing.assert_array_equal(masks[1], [True, False, False, False, False])
  pd.testing.assert_frame_equal(frame, original)

  output_path = tmp_path / "forecast.csv"
  forecast_csv.write_csv_output(results, str(output_path), frame, "date", 2)
  exported = pd.read_csv(output_path, parse_dates=["date"])
  for _, group in exported.groupby("series"):
    assert group["date"].tolist() == list(pd.date_range("2026-01-06", periods=2))


def test_all_missing_column_is_rejected_before_forecast():
  frame = pd.DataFrame({"sales": [1.0, np.nan], "empty": [np.nan, np.nan]})
  model = preprocessing_model()

  with pytest.raises(ValueError, match="Column 'empty' has no observed values"):
    forecast_csv.forecast_series(model, frame, ["sales", "empty"], 2)

  model.forecast.assert_not_called()
