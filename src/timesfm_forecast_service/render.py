from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_forecast_chart(
  output_dir: str | Path,
  history: pd.Series,
  future_index: pd.DatetimeIndex,
  point: np.ndarray,
  q10: np.ndarray,
  q90: np.ndarray,
  title: str = "TimesFM forecast",
) -> str:
  """Render history, point forecast and 10-90% interval to a local PNG."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.dates as mdates
  import matplotlib.pyplot as plt

  destination = Path(output_dir)
  destination.mkdir(parents=True, exist_ok=True)
  output = destination / "forecast.png"

  actual_color, forecast_color = "#2a78d6", "#1baf7a"
  surface, muted, grid, base = "#fcfcfb", "#898781", "#e1e0d9", "#c3c2b7"
  fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=surface)
  ax.set_facecolor(surface)

  future_line_x = pd.DatetimeIndex([history.index[-1]]).append(future_index)
  future_line_y = np.concatenate([[history.iloc[-1]], point])
  ax.fill_between(
    future_index, q10, q90, color=forecast_color, alpha=0.15,
    linewidth=0, label="80% prediction interval")
  ax.plot(
    history.index, history.to_numpy(), color=actual_color,
    linewidth=2, label="Actual")
  ax.plot(
    future_line_x, future_line_y, color=forecast_color, linewidth=2,
    marker="o", markersize=6, markeredgecolor=surface,
    markeredgewidth=2, label="Forecast")

  boundary = history.index[-1] + (future_index[0] - history.index[-1]) / 2
  ax.axvline(boundary, color=base, linewidth=1)
  ax.text(
    boundary, ax.get_ylim()[1], " forecast start ", ha="left", va="top",
    fontsize=9, color=muted)
  ax.yaxis.grid(True, color=grid, linewidth=1)
  ax.set_axisbelow(True)
  for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
  ax.spines["bottom"].set_color(base)
  ax.tick_params(colors=muted, length=0)
  ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
  ax.set_title(title, fontsize=12, color="#0b0b0b", loc="left", pad=12)
  handles, _ = ax.get_legend_handles_labels()
  lines = [handle for handle in handles if isinstance(handle, plt.Line2D)]
  bands = [handle for handle in handles if not isinstance(handle, plt.Line2D)]
  ax.legend(handles=lines + bands, frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(output, dpi=150, facecolor=surface)
  plt.close(fig)
  return str(output)
