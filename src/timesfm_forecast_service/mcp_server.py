from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .forecast_service import ForecastService, SeriesSpec


mcp = FastMCP("timesfm-forecast")
service = ForecastService()


@mcp.tool()
def inspect_dataset(
  path: str,
  query: str | None = None,
  max_values: int = 20,
) -> dict[str, Any]:
  """检查 Excel/CSV 的 sheet、字段、日期范围和维度合法值，并根据用户问题返回候选预测序列。

  在预测前调用。query 应包含用户原始意图，例如“预测 Glad 未来 6 个月销售额”。
  candidate_series 是经过文件内容确认的预测规格候选；不要猜测不存在的字段或筛选值。
  """
  return service.inspect_dataset(path, query, max_values)


@mcp.tool()
def run_forecast(
  path: str,
  spec: SeriesSpec,
  horizon: int = 6,
  request_id: str | None = None,
  output_dir: str | None = None,
  with_chart: bool = True,
) -> dict[str, Any]:
  """执行完整且可重试的预测：过滤聚合、校验、留出回测、TimesFM 预测和产物生成。

  spec 必须来自 inspect_dataset 确认后的字段和值，包含 sheet、time_col、value_col、filters、aggregation、frequency。
  相同 request_id 和规格会复用已完成结果。返回数据摘要、MAPE、80% 区间覆盖率、未来 point/q10/q90、质量等级、警告和 artifacts。
  """
  return service.run_forecast(
    path=path,
    spec=spec,
    horizon=horizon,
    request_id=request_id,
    output_dir=output_dir,
    with_chart=with_chart,
  )


def main():
  mcp.run()

