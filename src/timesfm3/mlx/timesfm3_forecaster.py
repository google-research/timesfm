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
    """Runs inference on a batch of series.

    Each context is a 1D univariate series or a 2D ``(num_variates, context)``
    multivariate series, optionally with per-series past-only and past-future
    covariates. A 1D input yields ``forecast`` of shape ``(horizon,)``; a 2D
    input yields ``(num_variates, horizon)``, matching the torch backend.
    """
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

    n = len(contexts)
    ids = list(ts_ids) if ts_ids is not None else [None] * n
    po_list = past_only_covariates if past_only_covariates is not None else [None] * n
    pf_list = (
      past_future_covariates if past_future_covariates is not None else [None] * n
    )
    median_idx = self.config.median_quantile_index
    cap = self.config.max_context_length

    def _shape_output(raw, num_target, was_1d, ctx_2d):
      # raw: (num_variates, horizon, num_quantiles). Keep only the target rows.
      tgt = raw[:num_target]
      if sort_quantiles:
        tgt = np.sort(tgt, axis=-1)
      forecast = np.array(tgt[..., median_idx])  # (num_target, horizon)
      quantiles = np.array(tgt) if return_quantiles else None
      if make_positive:
        # Match torch: clamp a variate only when its own input is nonnegative.
        nonneg = (ctx_2d >= 0).all(axis=1)
        for r in range(num_target):
          if nonneg[r]:
            forecast[r] = np.maximum(forecast[r], 0.0)
            if quantiles is not None:
              quantiles[r] = np.maximum(quantiles[r], 0.0)
      if was_1d:
        forecast = forecast[0]
        quantiles = quantiles[0] if quantiles is not None else None
      return forecast, quantiles

    results: list[ForecastOutput | None] = [None] * n
    has_cov = any(po is not None for po in po_list) or any(
      pf is not None for pf in pf_list
    )
    all_univariate = all(np.ndim(c) == 1 for c in contexts)

    if not has_cov and all_univariate:
      # Fast path: univariate, no covariates. Group by length so each group runs
      # through a single batched forward pass. decode() is per-series independent
      # (running stats, RevIN, detrending are per row), so this is numerically
      # identical to looping but scales throughput near-linearly.
      arrs = [np.asarray(c, dtype=np.float32).reshape(-1)[-cap:] for c in contexts]
      groups: dict[int, list[int]] = {}
      for i, a in enumerate(arrs):
        groups.setdefault(a.shape[0], []).append(i)
      for _length, idxs in groups.items():
        batch = mx.array(np.stack([arrs[i] for i in idxs]))[:, None, :]
        logits = np.array(self.model.decode(batch, horizon))  # (B, 1, h, q)
        for bi, i in enumerate(idxs):
          forecast, quantiles = _shape_output(logits[bi], 1, True, arrs[i][None, :])
          results[i] = ForecastOutput(
            ts_id=ids[i], forecast=forecast, quantiles=quantiles
          )
    else:
      # General path: multivariate targets and/or covariates, one series at a time.
      for i, c in enumerate(contexts):
        was_1d = np.ndim(c) == 1
        tgt = np.atleast_2d(np.asarray(c, dtype=np.float32))  # (u, ctx)
        ctx_len = tgt.shape[-1]
        po = po_list[i]
        pf = pf_list[i]
        po = np.atleast_2d(np.asarray(po, dtype=np.float32)) if po is not None else None
        pf = np.atleast_2d(np.asarray(pf, dtype=np.float32)) if pf is not None else None
        # global_context truncation, keeping covariate windows aligned with the
        # target window (as the torch Query.format does).
        if ctx_len > cap:
          tgt = tgt[:, -cap:]
          if po is not None:
            po = po[:, -cap:]
          if pf is not None:
            future_len = pf.shape[-1] - ctx_len
            pf = pf[:, -(cap + future_len) :]
          ctx_len = cap
        logits = np.array(
          self.model.decode(
            mx.array(tgt)[None],
            horizon,
            past_only_covariates=mx.array(po)[None] if po is not None else None,
            past_future_covariates=mx.array(pf)[None] if pf is not None else None,
          )
        )[0]  # (num_variates, h, q)
        forecast, quantiles = _shape_output(logits, tgt.shape[0], was_1d, tgt)
        results[i] = ForecastOutput(
          ts_id=ids[i], forecast=forecast, quantiles=quantiles
        )

    for r in results:
      yield r
