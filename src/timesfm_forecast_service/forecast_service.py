from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .model_runtime import TimesFMPredictor
from .render import save_forecast_chart


MIN_POINTS_HARD = 12
MIN_POINTS_RECOMMENDED = 32
MAX_HORIZON = 128


class SeriesSpec(BaseModel):
  sheet: str | None = None
  time_col: str
  value_col: str
  filters: dict[str, str | list[str]] = Field(default_factory=dict)
  aggregation: Literal["sum", "mean"] = "sum"
  frequency: str = "MS"


def _error(code: str, message: str, hint: str = "") -> dict[str, Any]:
  return {"ok": False, "error": {"code": code, "message": message, "hint": hint}}


def _atomic_json(path: Path, value: Any):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
  temporary.replace(path)


def _file_digest(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _safe_request_id(request_id: str | None) -> str:
  if request_id is None or not request_id.strip():
    return "fc_" + uuid.uuid4().hex[:12]
  value = request_id.strip()
  if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", value):
    raise ValueError("request_id 只能包含字母、数字、点、下划线和连字符")
  return value


def _candidate_time_columns(frame: pd.DataFrame) -> list[str]:
  candidates: list[str] = []
  for column in frame.columns:
    series = frame[column]
    name = str(column)
    named = any(token in name.casefold() for token in (
      "date", "period", "month", "time", "日期", "周期", "月份"))
    if pd.api.types.is_datetime64_any_dtype(series):
      candidates.append(name)
      continue
    if named:
      parsed = pd.to_datetime(series, errors="coerce")
      if parsed.notna().mean() >= 0.8 and parsed.nunique() >= 2:
        candidates.append(name)
  return candidates


def _candidate_numeric_columns(frame: pd.DataFrame) -> list[str]:
  return [str(column) for column in frame.columns
          if pd.api.types.is_numeric_dtype(frame[column])]


def _dimension_profiles(
  frame: pd.DataFrame, max_values: int,
) -> list[dict[str, Any]]:
  dimensions: list[dict[str, Any]] = []
  for column in frame.columns:
    series = frame[column].dropna()
    if (series.empty or pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)):
      continue
    values = series.astype(str).str.strip()
    unique = values[values.ne("")].drop_duplicates()
    if 0 < len(unique) <= 100:
      dimensions.append({
        "name": str(column),
        "n_unique": int(len(unique)),
        "sample_values": unique.head(max_values).tolist(),
      })
  return dimensions


def _preferred_value_column(columns: list[str], query: str) -> str | None:
  if not columns:
    return None
  query_lower = query.casefold()
  preferences: list[str] = []
  if "销售额" in query or "sales value" in query_lower or "revenue" in query_lower:
    preferences.extend(("sales_value", "sales value", "revenue", "销售额"))
  if "销量" in query or "units" in query_lower:
    preferences.extend(("units", "volume", "销量"))
  for preference in preferences:
    for column in columns:
      if preference in column.casefold():
        return column
  return columns[0]


class ForecastService:
  """Stateless forecasting interface used by the MCP adapter."""

  def __init__(self, predictor=None):
    self.predictor = predictor or TimesFMPredictor()

  def inspect_dataset(
    self, path: str, query: str | None = None, max_values: int = 20,
  ) -> dict[str, Any]:
    try:
      source = Path(path)
      if not source.is_file():
        return _error("FILE_NOT_FOUND", f"文件不存在: {path}")
      query = (query or "").strip()
      max_values = max(1, min(int(max_values), 50))
      frames: list[tuple[str, pd.DataFrame]] = []
      if source.suffix.casefold() in (".xlsx", ".xls"):
        with pd.ExcelFile(source) as workbook:
          frames = [
            (sheet, workbook.parse(sheet)) for sheet in workbook.sheet_names]
      elif source.suffix.casefold() == ".csv":
        frames = [("CSV", pd.read_csv(source))]
      else:
        return _error(
          "UNSUPPORTED_FILE", f"不支持的文件类型: {source.suffix}",
          "支持 .xlsx / .xls / .csv")

      profiles: list[dict[str, Any]] = []
      ranked: list[tuple[int, str, pd.DataFrame, list[str], list[str], list[dict[str, Any]]]] = []
      for sheet, frame in frames:
        time_columns = _candidate_time_columns(frame)
        numeric_columns = _candidate_numeric_columns(frame)
        dimensions = _dimension_profiles(frame, max_values)
        score = len(frame) + (100000 if time_columns and numeric_columns else 0)
        ranked.append((score, sheet, frame, time_columns, numeric_columns, dimensions))
        profiles.append({
          "name": sheet,
          "rows": int(len(frame)),
          "columns": [str(column) for column in frame.columns],
          "time_columns": time_columns,
          "numeric_columns": numeric_columns,
        })

      _, recommended_sheet, frame, time_columns, numeric_columns, dimensions = max(
        ranked, key=lambda item: item[0])
      date_range = None
      if time_columns:
        dates = pd.to_datetime(frame[time_columns[0]], errors="coerce").dropna()
        if not dates.empty:
          date_range = {
            "start": dates.min().strftime("%Y-%m-%d"),
            "end": dates.max().strftime("%Y-%m-%d"),
          }

      candidates: list[dict[str, Any]] = []
      query_lower = query.casefold()
      preferred_value = _preferred_value_column(numeric_columns, query)
      for dimension in dimensions:
        column = dimension["name"]
        values = frame[column].dropna().astype(str).str.strip().drop_duplicates()
        for value in values:
          if query_lower and value.casefold() in query_lower:
            candidate: dict[str, Any] = {
              "sheet": recommended_sheet,
              "dimension": column,
              "value": value,
              "time_col": time_columns[0] if time_columns else None,
              "value_col": preferred_value,
              "filters": {column: value},
              "aggregation": "sum",
              "frequency": "MS",
            }
            matches = frame[frame[column].astype(str).str.strip().eq(value)]
            for related in ("Manufacturer", "Brand", "Category"):
              if related in matches.columns and related != column:
                related_values = matches[related].dropna().astype(str).drop_duplicates().tolist()
                if len(related_values) == 1:
                  candidate[related.casefold()] = related_values[0]
            candidates.append(candidate)

      return {
        "ok": True,
        "source_path": str(source.resolve()),
        "recommended_data_sheet": recommended_sheet,
        "date_range": date_range,
        "sheets": profiles,
        "dimensions": dimensions,
        "candidate_series": candidates,
        "warnings": [] if candidates or not query else [
          "query_value_not_confirmed: 未在推荐数据表的低基数字段中确认查询对象，请从 dimensions 的合法值中选择"],
      }
    except Exception as exc:
      return _error("UNEXPECTED", f"{type(exc).__name__}: {exc}")

  def run_forecast(
    self,
    path: str,
    spec: SeriesSpec | dict[str, Any],
    horizon: int = 6,
    request_id: str | None = None,
    output_dir: str | None = None,
    with_chart: bool = True,
  ) -> dict[str, Any]:
    try:
      source = Path(path)
      if not source.is_file():
        return _error("FILE_NOT_FOUND", f"文件不存在: {path}")
      if not isinstance(spec, SeriesSpec):
        spec = SeriesSpec.model_validate(spec)
      if not 1 <= int(horizon) <= MAX_HORIZON:
        return _error("BAD_HORIZON", f"horizon 需在 1..{MAX_HORIZON}")
      horizon = int(horizon)
      run_id = _safe_request_id(request_id)
      run_directory = Path(output_dir) if output_dir else (
        source.parent / "artifacts" / "forecasts" / run_id)
      run_directory = run_directory.resolve()
      request_path = run_directory / "request.json"
      result_path = run_directory / "result.json"
      request_payload = {
        "request_id": run_id,
        "source_path": str(source.resolve()),
        "source_sha256": _file_digest(source),
        "series_spec": spec.model_dump(mode="json"),
        "horizon": horizon,
        "with_chart": bool(with_chart),
        "model": self.predictor.metadata(),
      }
      fingerprint = hashlib.sha256(json.dumps(
        request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
      request_payload["fingerprint"] = fingerprint

      if request_path.is_file():
        existing_request = json.loads(request_path.read_text(encoding="utf-8"))
        if existing_request.get("fingerprint") != fingerprint:
          return _error(
            "REQUEST_ID_CONFLICT",
            f"request_id '{run_id}' 已用于不同的预测规格",
            "为新规格使用新的 request_id")
        if result_path.is_file():
          result = json.loads(result_path.read_text(encoding="utf-8"))
          result["reused_request"] = True
          return result

      series, resolved_filters, source_rows, load_error = self._load_series(source, spec)
      if load_error:
        return load_error
      assert series is not None
      n_points = len(series)
      if n_points < MIN_POINTS_HARD:
        return _error(
          "INSUFFICIENT_CONTEXT",
          f"序列只有 {n_points} 个点 (<{MIN_POINTS_HARD})，不足以预测",
          "换更长历史、更低过滤粒度或更粗频率")
      if horizon > n_points // 2 or n_points - horizon < MIN_POINTS_HARD:
        return _error(
          "INSUFFICIENT_CONTEXT",
          f"{n_points} 个历史点不足以留出 {horizon} 个点并保留至少 {MIN_POINTS_HARD} 个训练点",
          "减小 horizon 或使用更长的历史序列")

      warnings: list[str] = []
      if n_points < MIN_POINTS_RECOMMENDED:
        warnings.append(
          f"context_below_recommended: 序列仅 {n_points} 个点 (<{MIN_POINTS_RECOMMENDED})，季节性识别不可靠")

      training = series.iloc[:-horizon].to_numpy(np.float32)
      actual = series.iloc[-horizon:]
      bt_point, bt_q10, bt_q90 = self.predictor.predict(training, horizon)
      actual_values = actual.to_numpy(dtype=float)
      nonzero = actual_values != 0
      if not nonzero.any():
        return _error("ZERO_HOLDOUT", "回测区间实际值全部为 0，无法计算 MAPE")
      mape = float(np.mean(np.abs(
        (actual_values[nonzero] - bt_point[nonzero]) / actual_values[nonzero])) * 100)
      coverage = float(np.mean(
        (actual_values >= bt_q10) & (actual_values <= bt_q90)) * 100)
      grade = (
        "excellent" if mape < 10 else "good" if mape < 20
        else "weak" if mape < 50 else "unreliable")

      point, q10, q90 = self.predictor.predict(
        series.to_numpy(np.float32), horizon)
      frequency = series.index.freqstr or spec.frequency
      future_index = pd.date_range(
        series.index[-1] + pd.tseries.frequencies.to_offset(frequency),
        periods=horizon, freq=frequency)
      _atomic_json(request_path, request_payload)

      subject = ", ".join(
        f"{key}={value}" for key, value in resolved_filters.items()) or spec.value_col
      artifacts: list[dict[str, Any]] = []
      if with_chart:
        chart_path = save_forecast_chart(
          run_directory, series, future_index, point, q10, q90,
          title=f"{subject} forecast")
        artifacts.append({
          "kind": "forecast_chart",
          "title": f"{subject} forecast chart",
          "mime_type": "image/png",
          "path": chart_path,
        })
      artifacts.append({
        "kind": "forecast_result",
        "title": f"{subject} forecast result",
        "mime_type": "application/json",
        "path": str(result_path),
      })

      result = {
        "ok": True,
        "request_id": run_id,
        "reused_request": False,
        "series_spec": {**spec.model_dump(mode="json"), "filters": resolved_filters},
        "series_summary": {
          "source_rows": source_rows,
          "n_points": n_points,
          "start": series.index[0].strftime("%Y-%m-%d"),
          "end": series.index[-1].strftime("%Y-%m-%d"),
          "head": self._rows(series.head(3)),
          "tail": self._rows(series.tail(3)),
        },
        "backtest": {
          "horizon": horizon,
          "mape": round(mape, 2),
          "coverage80": round(coverage, 1),
          "rows": [{
            "period": date.strftime("%Y-%m-%d"),
            "actual": round(float(observed), 2),
            "point": round(float(predicted), 2),
            "q10": round(float(lower), 2),
            "q90": round(float(upper), 2),
            "error_pct": None if observed == 0 else round(
              abs(float(observed) - float(predicted)) / abs(float(observed)) * 100, 1),
            "in_80pi": bool(lower <= observed <= upper),
          } for date, observed, predicted, lower, upper in zip(
            actual.index, actual_values, bt_point, bt_q10, bt_q90)],
        },
        "forecast": {
          "horizon": horizon,
          "rows": [{
            "period": date.strftime("%Y-%m-%d"),
            "point": round(float(predicted), 2),
            "q10": round(float(lower), 2),
            "q90": round(float(upper), 2),
          } for date, predicted, lower, upper in zip(future_index, point, q10, q90)],
        },
        "quality": {
          "grade": grade,
          "mape": round(mape, 2),
          "coverage80": round(coverage, 1),
        },
        "warnings": warnings,
        "model": self.predictor.metadata(),
        "artifacts": artifacts,
      }
      _atomic_json(result_path, result)
      return result
    except ValueError as exc:
      return _error("INVALID_ARGUMENT", str(exc))
    except Exception as exc:
      return _error("UNEXPECTED", f"{type(exc).__name__}: {exc}")

  def _load_series(
    self, source: Path, spec: SeriesSpec,
  ) -> tuple[pd.Series | None, dict[str, Any], int, dict[str, Any] | None]:
    if source.suffix.casefold() in (".xlsx", ".xls"):
      with pd.ExcelFile(source) as workbook:
        sheet = spec.sheet or workbook.sheet_names[0]
        if sheet not in workbook.sheet_names:
          return None, {}, 0, _error(
            "SHEET_NOT_FOUND", f"sheet '{sheet}' 不存在",
            f"可用 sheets: {workbook.sheet_names}")
        frame = workbook.parse(sheet)
    elif source.suffix.casefold() == ".csv":
      frame = pd.read_csv(source)
    else:
      return None, {}, 0, _error(
        "UNSUPPORTED_FILE", f"不支持的文件类型: {source.suffix}")

    required = [spec.time_col, spec.value_col, *spec.filters.keys()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
      return None, {}, 0, _error(
        "COLUMN_NOT_FOUND", f"列不存在: {missing}",
        f"现有列: {[str(column) for column in frame.columns]}")

    resolved_filters: dict[str, Any] = {}
    for column, requested in spec.filters.items():
      requested_values = requested if isinstance(requested, list) else [requested]
      normalized = frame[column].dropna().astype(str).str.strip()
      selected: list[str] = []
      mask = pd.Series(False, index=frame.index)
      for requested_value in requested_values:
        matches = normalized[
          normalized.str.casefold().eq(str(requested_value).strip().casefold())]
        exact_values = matches.drop_duplicates().tolist()
        if not exact_values:
          candidates = normalized[
            normalized.str.casefold().str.contains(
              re.escape(str(requested_value).strip().casefold()), regex=True)
          ].drop_duplicates().head(10).tolist()
          return None, {}, 0, _error(
            "FILTER_VALUE_NOT_FOUND",
            f"列 '{column}' 中不存在筛选值 '{requested_value}'",
            f"相近值: {candidates}")
        selected.extend(exact_values)
        mask |= frame[column].astype(str).str.strip().isin(exact_values)
      frame = frame[mask]
      resolved_filters[column] = selected if isinstance(requested, list) else selected[0]
    if frame.empty:
      return None, resolved_filters, 0, _error("EMPTY_AFTER_FILTER", "过滤后没有数据")
    source_rows = len(frame)

    dates = pd.to_datetime(frame[spec.time_col], errors="coerce")
    values = pd.to_numeric(frame[spec.value_col], errors="coerce")
    clean = pd.DataFrame({"date": dates, "value": values}).dropna()
    if clean.empty:
      return None, resolved_filters, source_rows, _error(
        "NO_USABLE_VALUES", "时间列或数值列没有可用数据")
    indexed = clean.set_index("date")["value"].sort_index()
    if spec.aggregation == "mean":
      series = indexed.resample(spec.frequency).mean()
    else:
      series = indexed.resample(spec.frequency).sum(min_count=1)
    if series.isna().any():
      missing_periods = [
        date.strftime("%Y-%m-%d") for date in series[series.isna()].index[:12]]
      return None, resolved_filters, source_rows, _error(
        "IRREGULAR_SERIES", "聚合后的时间序列存在缺口",
        f"缺失期间: {missing_periods}")
    return series.astype(float), resolved_filters, source_rows, None

  @staticmethod
  def _rows(series: pd.Series) -> list[dict[str, Any]]:
    return [{
      "period": date.strftime("%Y-%m-%d"),
      "value": round(float(value), 2),
    } for date, value in series.items()]


