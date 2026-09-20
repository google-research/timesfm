from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from timesfm_forecast_service.forecast_service import ForecastService, SeriesSpec


class FakePredictor:
  def __init__(self):
    self.calls = 0

  def predict(self, context: np.ndarray, horizon: int):
    self.calls += 1
    point = np.repeat(float(context[-1]), horizon)
    return point, point * 0.9, point * 1.1

  def metadata(self):
    return {"name": "fake-timesfm", "checkpoint": "test"}


def write_workbook(path: Path, periods: int = 24):
  rows = []
  for period in pd.date_range("2025-01-01", periods=periods, freq="MS"):
    for brand, manufacturer in (("Glad", "Clorox"), ("Blue Moon", "Blue Moon")):
      for channel in ("E-commerce", "Hyper/Super"):
        rows.append({
          "Period": period,
          "Brand": brand,
          "Manufacturer": manufacturer,
          "Channel": channel,
          "Category": "Trash Bags" if brand == "Glad" else "Laundry Detergent",
          "Sales_Value_RMB": 1000.0 + period.month * 10 + (100 if channel == "E-commerce" else 0),
        })
  with pd.ExcelWriter(path, engine="openpyxl") as writer:
    pd.DataFrame({"README": ["Synthetic test workbook"]}).to_excel(
      writer, sheet_name="00_README", index=False)
    pd.DataFrame(rows).to_excel(writer, sheet_name="01_NIQ_Style_Raw", index=False)


class ForecastServiceTest(unittest.TestCase):
  def test_inspect_dataset_finds_brand_and_recommends_fact_sheet(self):
    with tempfile.TemporaryDirectory() as tmp:
      workbook = Path(tmp) / "niq.xlsx"
      write_workbook(workbook)
      service = ForecastService(FakePredictor())

      result = service.inspect_dataset(str(workbook), query="预测 Glad 销售额")

      self.assertTrue(result["ok"], result)
      self.assertEqual(result["recommended_data_sheet"], "01_NIQ_Style_Raw")
      self.assertEqual(result["date_range"], {"start": "2025-01-01", "end": "2026-12-01"})
      self.assertTrue(any(
        candidate["dimension"] == "Brand" and candidate["value"] == "Glad"
        for candidate in result["candidate_series"]
      ))

  def test_run_forecast_is_atomic_persistent_and_idempotent(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      workbook = root / "niq.xlsx"
      write_workbook(workbook)
      predictor = FakePredictor()
      service = ForecastService(predictor)
      spec = SeriesSpec(
        sheet="01_NIQ_Style_Raw",
        time_col="Period",
        value_col="Sales_Value_RMB",
        filters={"Brand": "Glad"},
        aggregation="sum",
        frequency="MS",
      )

      result = service.run_forecast(
        path=str(workbook), spec=spec, horizon=6, request_id="glad-six-months")

      self.assertTrue(result["ok"], result)
      self.assertNotIn("series_id", result)
      self.assertEqual(result["series_summary"]["n_points"], 24)
      self.assertEqual(len(result["backtest"]["rows"]), 6)
      self.assertEqual(len(result["forecast"]["rows"]), 6)
      self.assertEqual(predictor.calls, 2)
      artifact_paths = {item["kind"]: Path(item["path"]) for item in result["artifacts"]}
      self.assertTrue(artifact_paths["forecast_chart"].is_file())
      self.assertTrue(artifact_paths["forecast_result"].is_file())
      self.assertTrue((artifact_paths["forecast_result"].parent / "request.json").is_file())

      reused = service.run_forecast(
        path=str(workbook), spec=spec, horizon=6, request_id="glad-six-months")

      self.assertTrue(reused["ok"], reused)
      self.assertTrue(reused["reused_request"])
      self.assertEqual(predictor.calls, 2)

  def test_same_request_id_rejects_a_different_forecast(self):
    with tempfile.TemporaryDirectory() as tmp:
      workbook = Path(tmp) / "niq.xlsx"
      write_workbook(workbook)
      service = ForecastService(FakePredictor())
      spec = SeriesSpec(
        sheet="01_NIQ_Style_Raw",
        time_col="Period",
        value_col="Sales_Value_RMB",
        filters={"Brand": "Glad"},
      )
      first = service.run_forecast(str(workbook), spec, 3, "same-request")
      self.assertTrue(first["ok"], first)

      conflict = service.run_forecast(str(workbook), spec, 6, "same-request")

      self.assertFalse(conflict["ok"])
      self.assertEqual(conflict["error"]["code"], "REQUEST_ID_CONFLICT")


if __name__ == "__main__":
  unittest.main()
