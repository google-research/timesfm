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
import math
from typing import Iterator

import mlx.core as mx
import numpy as np

from . import model as mlx_model_lib

# Below this, a series is treated as constant and z-normalization uses sigma=1
# (matches the torch backend's _SIGMA_THRESHOLD).
_SIGMA_THRESHOLD = 1e-7


def _znorm_stats(arr: np.ndarray) -> tuple[float, float]:
  """(mean, std) for z-normalization, ignoring NaNs; matches the torch backend."""
  mu = float(np.nanmean(arr))
  sigma = float(np.nanstd(arr))
  if not np.isfinite(mu):
    mu = 0.0
  if not np.isfinite(sigma) or sigma < _SIGMA_THRESHOLD:
    sigma = 1.0
  return mu, sigma


def _znorm_rows(arr: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
  """Z-normalize each row by its own stats; return the normalized array and stats."""
  stats = [_znorm_stats(row) for row in arr]
  out = np.stack([(row - mu) / sigma for row, (mu, sigma) in zip(arr, stats)])
  return out.astype(np.float32), stats


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
    if padding_mode not in ("none", "edge"):
      raise ValueError(f"Unknown padding_mode: {padding_mode!r}")
    if len(contexts) == 0:
      return

    n = len(contexts)
    ids = list(ts_ids) if ts_ids is not None else [None] * n
    po_list = past_only_covariates if past_only_covariates is not None else [None] * n
    pf_list = (
      past_future_covariates if past_future_covariates is not None else [None] * n
    )
    median_idx = self.config.median_quantile_index
    cap = self.config.max_context_length

    def _decode_np(tgt_2d, po_2d, pf_2d):
      # One series (2D target + optional 2D covariates) -> (num_variates, h, q).
      return np.array(
        self.model.decode(
          mx.array(tgt_2d)[None],
          horizon,
          past_only_covariates=mx.array(po_2d)[None] if po_2d is not None else None,
          past_future_covariates=mx.array(pf_2d)[None] if pf_2d is not None else None,
        )
      )[0]

    def _series_logits(tgt_2d, po_2d, pf_2d):
      # Final (num_variates, h, q) logits: sorted (if requested), and symmetric-
      # averaged when asked. Symmetric averaging runs the series and its negation
      # (and negated covariates), sorts each, then averages the quantile-reversed
      # negative run: (pos - neg[..., ::-1]) / 2, exactly as the torch backend.
      if use_symmetric_averaging:
        pos = _decode_np(tgt_2d, po_2d, pf_2d)
        neg = _decode_np(
          -tgt_2d,
          None if po_2d is None else -po_2d,
          None if pf_2d is None else -pf_2d,
        )
        if sort_quantiles:
          pos = np.sort(pos, axis=-1)
          neg = np.sort(neg, axis=-1)
        return (pos - neg[..., ::-1]) / 2
      out = _decode_np(tgt_2d, po_2d, pf_2d)
      return np.sort(out, axis=-1) if sort_quantiles else out

    def _shape(final, num_target, was_1d, ctx_2d):
      # final: (num_variates, h, q), already sorted / averaged. Keep target rows.
      tgt = final[:num_target]
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

    if (
      not has_cov
      and all_univariate
      and not use_symmetric_averaging
      and not use_znorm
      and padding_mode == "none"
    ):
      # Fast path: plain univariate, no covariates / symmetric averaging / znorm /
      # padding. Group by length so each group runs through a single batched
      # forward pass. decode() is per-series independent (running stats, RevIN,
      # detrending are per row), so this is numerically identical to looping but
      # scales throughput.
      arrs = [np.asarray(c, dtype=np.float32).reshape(-1)[-cap:] for c in contexts]
      groups: dict[int, list[int]] = {}
      for i, a in enumerate(arrs):
        groups.setdefault(a.shape[0], []).append(i)
      for _length, idxs in groups.items():
        batch = mx.array(np.stack([arrs[i] for i in idxs]))[:, None, :]
        logits = np.array(self.model.decode(batch, horizon))  # (B, 1, h, q)
        if sort_quantiles:
          logits = np.sort(logits, axis=-1)
        for bi, i in enumerate(idxs):
          forecast, quantiles = _shape(logits[bi], 1, True, arrs[i][None, :])
          results[i] = ForecastOutput(
            ts_id=ids[i], forecast=forecast, quantiles=quantiles
          )
    else:
      # General path: multivariate targets, covariates, symmetric averaging,
      # z-normalization and/or padding, one series at a time.
      opl = self.model.config.output_patch_len
      for i, c in enumerate(contexts):
        was_1d = np.ndim(c) == 1
        tgt_orig = np.atleast_2d(np.asarray(c, dtype=np.float32))  # (u, ctx)
        tgt = tgt_orig
        po = po_list[i]
        pf = pf_list[i]
        po = np.atleast_2d(np.asarray(po, dtype=np.float32)) if po is not None else None
        pf = np.atleast_2d(np.asarray(pf, dtype=np.float32)) if pf is not None else None

        # z-normalize on the full series (as torch does, before truncation): each
        # target and covariate row by its own stats. Only target stats are kept,
        # to un-normalize the target forecasts afterwards.
        tgt_stats = None
        if use_znorm:
          tgt, tgt_stats = _znorm_rows(tgt)
          if po is not None:
            po, _ = _znorm_rows(po)
          if pf is not None:
            pf, _ = _znorm_rows(pf)

        # global_context truncation, keeping covariate windows aligned with the
        # target window (as the torch Query.format does).
        ctx_len = tgt.shape[-1]
        if ctx_len > cap:
          tgt = tgt[:, -cap:]
          if po is not None:
            po = po[:, -cap:]
          if pf is not None:
            future_len = pf.shape[-1] - ctx_len
            pf = pf[:, -(cap + future_len) :]

        # padding_mode="edge": extend the past-future covariate to the
        # patch-rounded horizon (global_horizon) by repeating its last value, so
        # the model sees a covariate for every decoded step. The output is still
        # trimmed back to the requested horizon below.
        if padding_mode == "edge" and pf is not None:
          global_horizon = math.ceil(horizon / opl) * opl
          pad_len = global_horizon - horizon
          if pad_len > 0:
            pf = np.pad(pf, [(0, 0), (0, pad_len)], mode="edge")

        final = _series_logits(tgt, po, pf)  # (num_variates, decoded_h, q)
        final = final[:, :horizon, :]  # trim edge-padded horizon back to requested
        if use_znorm:
          for r in range(tgt.shape[0]):
            mu, sigma = tgt_stats[r]
            final[r] = final[r] * sigma + mu

        forecast, quantiles = _shape(final, tgt.shape[0], was_1d, tgt_orig)
        results[i] = ForecastOutput(
          ts_id=ids[i], forecast=forecast, quantiles=quantiles
        )

    for r in results:
      yield r
