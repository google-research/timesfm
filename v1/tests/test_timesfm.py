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


from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import timesfm


def create_sample_dataframe(
    start_date: datetime, end_date: datetime, freq: str = "D"
) -> pd.DataFrame:
    """
    Create a sample DataFrame with time series data.

    Args:
        start_date (datetime): Start date of the time series.
        end_date (datetime): End date of the time series.
        freq (str): Frequency of the time series (default: "D" for daily).

    Returns:
        pd.DataFrame: DataFrame with columns 'unique_id', 'ds', and 'ts'.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    ts_data = np.random.randn(len(date_range))
    df = pd.DataFrame({"unique_id": "ts-1", "ds": date_range, "ts": ts_data})
    return df


@pytest.mark.parametrize("context_length", [128, 256, 512])
@pytest.mark.parametrize("prediction_length", [96, 128, 256])
@pytest.mark.parametrize("freq", ["D", "H", "W"])
def test_timesfm_forecast_on_df(
    context_length: int,
    prediction_length: int,
    freq: str,
) -> None:
    model = timesfm.TimesFm(
        context_len=context_length,
        horizon_len=prediction_length,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
    )
    model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=context_length)
    input_df = create_sample_dataframe(start_date, end_date, freq)

    forecast_df = model.forecast_on_df(
        inputs=input_df,
        freq=freq,
        value_name="ts",
        num_jobs=-1,
    )

    assert (
        len(forecast_df) == prediction_length
    ), f"Expected forecast length of {prediction_length}, but got {len(forecast_df)}"
    assert (
        "timesfm" in forecast_df.columns
    ), "Forecast DataFrame should contain 'timesfm' column"

    last_input_date = input_df["ds"].max()
    first_forecast_date = forecast_df["ds"].min()
    expected_first_forecast_date = last_input_date + pd.Timedelta(1, unit=freq)
    assert (
        first_forecast_date == expected_first_forecast_date
    ), f"Forecast should start from {expected_first_forecast_date}, but starts from {first_forecast_date}"

    print(
        f"Successful forecast with context_length={context_length}, prediction_length={prediction_length}, freq={freq}"
    )
