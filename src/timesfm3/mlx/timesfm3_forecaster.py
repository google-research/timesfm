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

"""High-level MLX TimesFM3 forecaster.

Mirrors the interface of the PyTorch ``TimesFM3Forecaster`` (``from_pretrained`` / ``predict`` /
``predict_batch`` / ``_ModelConfig`` / ``ForecastOutput``) so the two backends are drop-in
compatible for the univariate (target-only) forecasting path.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

import mlx.core as mx
import numpy as np

from . import model as mlx_model_lib


@dataclasses.dataclass
class _ModelConfig:
  """Configuration for an MLX TimesFM3 forecaster."""

  # Path to a checkpoint directory or a Hugging Face repo id.
  checkpoint_path: str = "google/timesfm-3.0-pytorch"
  # Batch size to use for inference.
  per_core_batch_size: int = 4
  # Median quantile index for the point forecast.
  median_quantile_index: int = 4
  # mx.compile the forward pass (fuses kernels, cuts dispatch overhead).
  compile: bool = True
  # Longest context fed to the model. Contexts beyond this are truncated to
  # their most recent `max_context_length` points before decode, matching the
  # torch backend's `global_context` cap (`_MAX_CONTEXT_LENGTH`).
  max_context_length: int = 15360
  # Hugging Face download options.
  cache_dir: str | None = None
  revision: str | None = None
  token: str | None = None
  local_files_only: bool = False
  force_download: bool = False


# Public alias, matching the PyTorch backend.
ModelConfig = _ModelConfig


@dataclasses.dataclass(frozen=True)
class ForecastOutput:
  """Structured output of a forecast on a single time series."""

  # Optional identifier for this time series. Only set if provided on input.
  ts_id: str | None = None
  # Point forecast (median quantile) for the given horizon.
  forecast: np.ndarray | None = None
  # Full quantile forecasts. Optional.
  quantiles: np.ndarray | None = None


class TimesFM3Forecaster:
  """MLX TimesFM3 forecaster."""

  def __init__(self, config: _ModelConfig | None = None, **kwargs):
    self.config = config or _ModelConfig(**kwargs)
    self.model: mlx_model_lib.TimesFM3Mlx | None = None
    self._init_model()

  @classmethod
  def from_pretrained(
    cls, pretrained_model_name_or_path: str, **kwargs
  ) -> "TimesFM3Forecaster":
    """Build a forecaster from a checkpoint directory or Hugging Face repo id."""
    return cls(_ModelConfig(checkpoint_path=pretrained_model_name_or_path, **kwargs))

  def _init_model(self):
    """Initializes the MLX model and loads weights."""
    self.model = mlx_model_lib.TimesFM3Mlx.from_pretrained(
      self.config.checkpoint_path,
      compile=self.config.compile,
      cache_dir=self.config.cache_dir,
      revision=self.config.revision,
      token=self.config.token,
      local_files_only=self.config.local_files_only,
      force_download=self.config.force_download,
    )
    median_q_idx = self.config.median_quantile_index
    if median_q_idx >= self.model.config.num_quantiles:
      median_q_idx = self.model.config.num_quantiles // 2
    self.config = dataclasses.replace(self.config, median_quantile_index=median_q_idx)

  @property
  def context_length(self) -> int:
    return self.model.config.input_patch_len

  @property
  def global_context(self) -> int:
    """Longest context the model runs on; longer inputs are truncated to it."""
    return self.config.max_context_length

  def predict(
    self,
    context: np.ndarray,
    horizon: int,
    past_only_covariates: np.ndarray | None = None,
    past_future_covariates: np.ndarray | None = None,
    ts_id: str | None = None,
    return_quantiles: bool = False,
    use_symmetric_averaging: bool = False,
    make_positive: bool = False,
    sort_quantiles: bool = True,
    use_znorm: bool = False,
    padding_mode: str = "none",
  ) -> ForecastOutput:
    """Runs inference on a single time series (target-only path)."""
    results = list(
      self.predict_batch(
        contexts=[context],
        horizon=horizon,
        past_only_covariates=[past_only_covariates],
        past_future_covariates=[past_future_covariates],
        ts_ids=[ts_id] if ts_id is not None else None,
        return_quantiles=return_quantiles,
        use_symmetric_averaging=use_symmetric_averaging,
        make_positive=make_positive,
        sort_quantiles=sort_quantiles,
        use_znorm=use_znorm,
        padding_mode=padding_mode,
      )
    )
    return results[0]

  def predict_batch(
    self,
    contexts: list[np.ndarray],
    horizon: int,
    past_only_covariates: list[np.ndarray | None] | None = None,
    past_future_covariates: list[np.ndarray | None] | None = None,
    ts_ids: list[str] | None = None,
    return_quantiles: bool = False,
    use_symmetric_averaging: bool = False,
    make_positive: bool = False,
    sort_quantiles: bool = True,
    use_znorm: bool = False,
    padding_mode: str = "none",
  ) -> Iterator[ForecastOutput]:
    """Runs inference on a batch of univariate time series."""
    if any(c is not None for c in (past_only_covariates or [])) or any(
      c is not None for c in (past_future_covariates or [])
    ):
      raise NotImplementedError(
        "Covariate forecasting is not yet supported by the MLX backend; use the torch backend "
        "for covariates. Follow-up: add the multivariate/covariate path to the MLX model."
      )
    for name, flag in (
      ("use_symmetric_averaging", use_symmetric_averaging),
      ("use_znorm", use_znorm),
    ):
      if flag:
        raise NotImplementedError(
          f"{name}=True is not yet supported by the MLX backend."
        )
    if padding_mode != "none":
      raise NotImplementedError(
        f"padding_mode={padding_mode!r} is not yet supported by the MLX backend."
      )
    if not contexts:
      return

    ids = list(ts_ids) if ts_ids is not None else [None] * len(contexts)
    median_idx = self.config.median_quantile_index

    # Group same-length contexts so each group runs through a single batched forward pass.
    # decode() is per-series independent (running stats, RevIN, detrending are per-batch-row), so
    # batching is numerically identical to looping but scales throughput near-linearly.
    cap = self.config.max_context_length
    arrs = [np.asarray(c, dtype=np.float32).reshape(-1)[-cap:] for c in contexts]
    groups: dict[int, list[int]] = {}
    for i, a in enumerate(arrs):
      groups.setdefault(a.shape[0], []).append(i)

    results: list[ForecastOutput | None] = [None] * len(arrs)
    for _length, idxs in groups.items():
      batch = mx.array(np.stack([arrs[i] for i in idxs]))[:, None, :]  # (B, 1, length)
      logits = self.model.decode(batch, horizon)  # (B, 1, horizon, num_quantiles)
      for bi, i in enumerate(idxs):
        q = logits[bi, 0]  # (horizon, num_quantiles)
        if sort_quantiles:
          q = mx.sort(q, axis=-1)
        forecast = np.array(q[:, median_idx])
        if make_positive:
          forecast = np.maximum(forecast, 0.0)
        quantiles = None
        if return_quantiles:
          quantiles = np.array(q)
          if make_positive:
            quantiles = np.maximum(quantiles, 0.0)
        results[i] = ForecastOutput(
          ts_id=ids[i], forecast=forecast, quantiles=quantiles
        )

    for r in results:
      yield r
