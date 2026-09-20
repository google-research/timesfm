"""Regression test for keeping generated forecasts out of Archer's upload tray."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from timesfm_forecast_service.render import save_forecast_chart


class ForecastOutputPathTest(unittest.TestCase):

  def test_chart_is_saved_in_run_artifact_directory(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      history_index = pd.date_range("2025-01-01", periods=3, freq="MS")
      history = pd.Series([100.0, 110.0, 120.0], index=history_index)
      future_index = pd.date_range("2025-04-01", periods=2, freq="MS")
      output_dir = root / "artifacts" / "forecasts" / "test-run"

      output = Path(save_forecast_chart(
        output_dir,
        history,
        future_index,
        np.array([125.0, 130.0]),
        np.array([115.0, 118.0]),
        np.array([135.0, 142.0]),
      ))

      self.assertEqual(output.parent, output_dir)
      self.assertTrue(output.is_file())
      self.assertEqual(list(root.glob("*_timesfm_forecast.png")), [])


if __name__ == "__main__":
  unittest.main()
