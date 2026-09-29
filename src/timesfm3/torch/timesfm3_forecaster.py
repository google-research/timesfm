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

"""Forecaster API wrapping a pretrained TimesFM3 PyTorch model."""

from __future__ import annotations

import dataclasses
import gc
import math
import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from . import configs, util
from . import model as torch_model_lib

_MAX_CONTEXT_LENGTH = 15360
_SIGMA_THRESHOLD: float = 1e-7
_GC_MEMORY_THRESHOLD: float = 0.9


@dataclasses.dataclass
class _ModelConfig:
  """Configuration for a PyTorch TimesFM3 forecaster."""

  # Path to checkpoint file (.pth or .safetensors) or Hugging Face repo ID.
  checkpoint_path: str = "google/timesfm-3.0-pytorch"

  # Batch size to use for inference.
  per_core_batch_size: int = 4

  # Input (context) patch length used by the model.
  input_patch_length: int = 32

  # Output (horizon) patch length used by the model.
  output_patch_length: int = 64

  # Quantiles to predict.
  quantiles: list[float] = dataclasses.field(
    default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  )

  # Median quantile index for the forecast.
  median_quantile_index: int = 4

  # Whether to use stitching.
  use_stitching: bool = True

  # Whether to use linear detrending.
  use_linear_detrending: bool = True

  # Linear detrending threshold.
  linear_detrending_threshold: float = 0.5

  # Whether to use iterative CPM RevIN refinement.
  use_iterative_cpm_revin: bool = True

  # Whether running stats freeze at context boundary.
  use_frozen_running_stats: bool = False

  # Whether to use variate attention.
  use_variate_attention: bool = True

  # Value clipping bound.
  value_clip: float = 1e20

  # Input transformation.
  input_transform: str = "identity"

  # Whether to use scaled_dot_product_attention.
  use_sdpa: bool = True

  # Optional PyTorch device string ('cuda', 'cpu', etc.).
  device: str | None = None

  # Optional configuration objects
  residual_block_config: configs.ResidualBlockConfig | None = None
  transformer_config: configs.StackedTransformersConfig | None = None

  # Hugging Face Hub download options
  cache_dir: str | None = None
  force_download: bool = False
  token: str | bool | None = None
  revision: str | None = None
  local_files_only: bool = False


# Public alias
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
  # Optional confidence diagnostics derived from quantile dispersion.
  diagnostics: ForecastDiagnostics | None = None


@dataclasses.dataclass(frozen=True)
class ForecastDiagnostics:
  """Confidence diagnostics computed across forecast horizons.

  The diagnostics are derived from the widest available central prediction
  interval. Endpoints are normalized independently of the returned quantile
  array so diagnostics remain meaningful when raw quantile outputs cross.
  """

  # Width of the prediction interval at each horizon step.
  interval_width: np.ndarray
  # Whether the raw interval endpoints were crossed before normalization.
  crossed_interval: np.ndarray
  # Width normalized by the absolute point forecast at each horizon step.
  relative_interval_width: np.ndarray
  # Interval width divided by the first horizon step width.
  width_growth: np.ndarray
  # Confidence implied by absolute interval width relative to forecast magnitude.
  magnitude_confidence: np.ndarray
  # Confidence implied by interval widening across the horizon.
  growth_confidence: np.ndarray
  # Coarse confidence bucket per horizon step: high, moderate, or low.
  confidence: np.ndarray
  # The quantile levels used as lower and upper bounds.
  lower_quantile: float
  upper_quantile: float


def forecast_confidence_diagnostics(
  forecast: np.ndarray,
  quantiles: np.ndarray,
  quantile_levels: list[float],
  eps: float = 1e-6,
  relative_width_moderate_threshold: float = 0.2,
  relative_width_low_threshold: float = 0.5,
  growth_moderate_threshold: float = 1.5,
  growth_low_threshold: float = 3.0,
) -> ForecastDiagnostics:
  """Computes simple confidence diagnostics from forecast quantiles.

  Args:
    forecast: Point forecast with shape ``(horizon,)`` or
      ``(num_variates, horizon)``.
    quantiles: Quantile forecasts with shape ``(horizon, num_quantiles)`` or
      ``(num_variates, horizon, num_quantiles)``.
    quantile_levels: Quantile levels corresponding to the final axis.
    eps: Minimum denominator for ratio metrics.
    relative_width_moderate_threshold: Relative interval width above which
      magnitude confidence is labelled moderate.
    relative_width_low_threshold: Relative interval width above which magnitude
      confidence is labelled low.
    growth_moderate_threshold: Width growth above which growth confidence is
      labelled moderate.
    growth_low_threshold: Width growth above which growth confidence is labelled
      low.

  Returns:
    ForecastDiagnostics with arrays matching the forecast shape.
  """
  forecast_arr = np.asarray(forecast, dtype=np.float32)
  quantile_arr = np.asarray(quantiles, dtype=np.float32)

  if quantile_arr.ndim < 2:
    raise ValueError("quantiles must include horizon and quantile dimensions.")
  if quantile_arr.shape[:-1] != forecast_arr.shape:
    raise ValueError(
      "quantiles shape before the final axis must match forecast shape: "
      f"{quantile_arr.shape[:-1]} != {forecast_arr.shape}."
    )
  if quantile_arr.shape[-1] < 2:
    raise ValueError("At least two quantile levels are required for diagnostics.")
  if len(quantile_levels) != quantile_arr.shape[-1]:
    raise ValueError(
      "quantile_levels length must match the quantile dimension: "
      f"{len(quantile_levels)} != {quantile_arr.shape[-1]}."
    )

  raw_lower_endpoint = quantile_arr[..., 0]
  raw_upper_endpoint = quantile_arr[..., -1]
  lower_endpoint = np.minimum(raw_lower_endpoint, raw_upper_endpoint)
  upper_endpoint = np.maximum(raw_lower_endpoint, raw_upper_endpoint)
  crossed_interval = raw_lower_endpoint > raw_upper_endpoint
  interval_width = upper_endpoint - lower_endpoint
  relative_interval_width = interval_width / np.maximum(
    np.abs(forecast_arr), eps
  )
  first_width = np.take(interval_width, indices=0, axis=-1)
  width_growth = interval_width / np.maximum(np.expand_dims(first_width, -1), eps)
  magnitude_confidence = np.full(interval_width.shape, "high", dtype=object)
  magnitude_confidence = np.where(
    relative_interval_width > relative_width_moderate_threshold,
    "moderate",
    magnitude_confidence,
  )
  magnitude_confidence = np.where(
    relative_interval_width > relative_width_low_threshold,
    "low",
    magnitude_confidence,
  )
  growth_confidence = np.full(interval_width.shape, "high", dtype=object)
  growth_confidence = np.where(
    width_growth > growth_moderate_threshold, "moderate", growth_confidence
  )
  growth_confidence = np.where(
    width_growth > growth_low_threshold, "low", growth_confidence
  )

  confidence_rank = {"high": 0, "moderate": 1, "low": 2}
  confidence_labels = np.array(["high", "moderate", "low"], dtype=object)
  magnitude_rank = np.vectorize(confidence_rank.__getitem__)(magnitude_confidence)
  growth_rank = np.vectorize(confidence_rank.__getitem__)(growth_confidence)
  confidence = confidence_labels[np.maximum(magnitude_rank, growth_rank)]

  return ForecastDiagnostics(
    interval_width=interval_width,
    crossed_interval=crossed_interval,
    relative_interval_width=relative_interval_width,
    width_growth=width_growth,
    magnitude_confidence=magnitude_confidence,
    growth_confidence=growth_confidence,
    confidence=confidence,
    lower_quantile=float(quantile_levels[0]),
    upper_quantile=float(quantile_levels[-1]),
  )


def try_gc(
  device: torch.device | str | None = None,
  gc_memory_threshold: float = _GC_MEMORY_THRESHOLD,
) -> None:
  """Trigger Python GC and empty CUDA cache if memory exceeds threshold."""
  if device is not None and torch.cuda.is_available():
    d = torch.device(device) if isinstance(device, str) else device
    if d.type == "cuda":
      allocated = torch.cuda.memory_allocated(d)
      total = torch.cuda.get_device_properties(d).total_memory
      if total > 0 and (allocated / total) > gc_memory_threshold:
        gc.collect()
        torch.cuda.empty_cache()
        return
  gc.collect()


def linear_interpolation(arr: np.ndarray) -> np.ndarray:
  """Performs linear interpolation to fill NaN values in a NumPy array."""
  was_1d = arr.ndim == 1
  arr2d = np.atleast_2d(arr)

  nan_mask = np.isnan(arr2d)
  if not np.any(nan_mask):
    return arr

  result = arr2d.copy()
  for r in range(result.shape[0]):
    if np.any(nan_mask[r]):
      row = result[r]
      valid_mask = ~nan_mask[r]
      nan_indices = nan_mask[r].nonzero()[0]
      non_nan_indices = valid_mask.nonzero()[0]
      non_nan_values = row[valid_mask]
      try:
        row[nan_mask[r]] = np.interp(nan_indices, non_nan_indices, non_nan_values)
      except ValueError:
        if non_nan_values.size > 0:
          mu = np.nanmean(row)
        else:
          mu = 0.0
        row[nan_mask[r]] = mu

  return result[0] if was_1d else result


def _znorm_stats(arr: np.ndarray) -> tuple[float, float]:
  """Returns (mean, std) for z-normalization, ignoring NaNs."""
  mu = float(np.nanmean(arr))
  sigma = float(np.nanstd(arr))
  if not np.isfinite(mu):
    mu = 0.0
  if not np.isfinite(sigma) or sigma < _SIGMA_THRESHOLD:
    sigma = 1.0
  return mu, sigma


def _is_nonnegative(arr: np.ndarray) -> bool | np.ndarray:
  """Returns True if `arr` has at least one finite value and all are >= 0."""
  was_1d = arr.ndim == 1
  arr2d = np.atleast_2d(arr)

  result = np.zeros(arr2d.shape[0], dtype=bool)
  for r in range(arr2d.shape[0]):
    valid = arr2d[r][~np.isnan(arr2d[r])]
    result[r] = valid.size > 0 and bool(np.all(valid >= 0))

  return bool(result[0]) if was_1d else result


@dataclasses.dataclass(frozen=True)
class _Query:
  """Represents a single formatted forecast query."""

  horizon: int
  targets: np.ndarray
  past_only_covariates: np.ndarray | None = None
  past_future_covariates: np.ndarray | None = None

  @property
  def context_length(self) -> int:
    return self.targets.shape[-1]

  def format(
    self, context_len: int
  ) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
  ]:
    """Formats and left-pads/truncates the query to context_len length."""
    targets = np.atleast_2d(self.targets)
    masks = np.zeros((self.context_length,), dtype=bool)
    past_only_covariates = (
      np.atleast_2d(self.past_only_covariates)
      if self.past_only_covariates is not None
      else None
    )
    past_future_covariates = (
      np.atleast_2d(self.past_future_covariates)
      if self.past_future_covariates is not None
      else None
    )

    if self.context_length > context_len:
      targets = targets[:, -context_len:]
      masks = masks[-context_len:]
      if past_only_covariates is not None:
        past_only_covariates = past_only_covariates[:, -context_len:]
      if past_future_covariates is not None:
        # Keep the covariate window aligned with the target window: the
        # future part of `past_future_covariates` may be shorter than
        # `self.horizon` (the patch-rounded horizon) when padding_mode is
        # "none", so slice by the covariate's own future length rather than
        # by `self.horizon`.
        future_len = past_future_covariates.shape[-1] - self.context_length
        past_future_covariates = past_future_covariates[
          :, -(context_len + future_len) :
        ]
    elif self.context_length < context_len:
      pad_len = context_len - self.context_length
      targets = np.pad(
        targets,
        [(0, 0), (pad_len, 0)],
        mode="constant",
        constant_values=0.0,
      )
      masks = np.pad(
        masks,
        [(pad_len, 0)],
        mode="constant",
        constant_values=True,
      )
      if past_only_covariates is not None:
        past_only_covariates = np.pad(
          past_only_covariates,
          [(0, 0), (pad_len, 0)],
          mode="constant",
          constant_values=0.0,
        )
      if past_future_covariates is not None:
        past_future_covariates = np.pad(
          past_future_covariates,
          [(0, 0), (pad_len, 0)],
          mode="constant",
          constant_values=0.0,
        )

    return (
      self.horizon,
      targets,
      masks,
      past_only_covariates,
      past_future_covariates,
    )


def _make_torch_model(
  config: _ModelConfig,
) -> torch_model_lib.TimesFM3Torch:
  """Builds a PyTorch model using ModelConfig."""
  resblock_config = (
    config.residual_block_config
    if config.residual_block_config is not None
    else configs.ResidualBlockConfig(
      hidden_dims=1280,
      output_dims=1280,
      use_bias=False,
      activation="relu",
    )
  )

  transformer_config = (
    config.transformer_config
    if config.transformer_config is not None
    else configs.StackedTransformersConfig(
      num_layers=20,
      transformer=configs.TransformerConfig(
        model_dims=1280,
        hidden_dims=1280,
        num_heads=16,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_rope_seq=True,
        use_rope_var=True,
        use_bias=False,
        ff_activation="relu",
        deterministic=True,
      ),
    )
  )

  t_model = torch_model_lib.TimesFM3Torch(
    input_patch_len=config.input_patch_length,
    output_patch_len=config.output_patch_length,
    quantiles=config.quantiles,
    use_variate_attention=config.use_variate_attention,
    value_clip=config.value_clip,
    input_transform=config.input_transform,
    use_stitching=config.use_stitching,
    use_linear_detrending=config.use_linear_detrending,
    linear_detrending_threshold=config.linear_detrending_threshold,
    use_iterative_cpm_revin=config.use_iterative_cpm_revin,
    use_frozen_running_stats=config.use_frozen_running_stats,
    residual_block_config=resblock_config,
    transformer_config=transformer_config,
  )
  t_model.eval()
  input_dim = 2 * (t_model.input_patch_len + t_model.output_patch_len)
  t_model.pre_transformer_resblock.set_input_dims(input_dim)
  return t_model


class TimesFM3Forecaster:
  """Forecaster wrapping a PyTorch TimesFM3 model for inference."""

  def __init__(self, config: _ModelConfig | None = None, **kwargs):
    if config is None:
      self.config = _ModelConfig(**kwargs)
    elif kwargs:
      self.config = dataclasses.replace(config, **kwargs)
    else:
      self.config = config
    self._init_model()

  @classmethod
  def from_pretrained(
    cls,
    pretrained_model_name_or_path: str = "google/timesfm-3.0-pytorch",
    device: str | None = None,
    **kwargs: Any,
  ) -> TimesFM3Forecaster:
    """Instantiates a TimesFM3Forecaster from a pretrained HF repo or directory."""
    config = _ModelConfig(
      checkpoint_path=pretrained_model_name_or_path,
      device=device,
      **kwargs,
    )
    return cls(config=config)

  @property
  def global_context(self) -> int:
    """Max context length rounded up to the nearest input patch boundary."""
    return (
      math.ceil(_MAX_CONTEXT_LENGTH / self.config.input_patch_length)
      * self.config.input_patch_length
    )

  def _init_model(self):
    """Initializes the PyTorch model and loads weights."""
    if self.config.device is not None:
      self.device = torch.device(self.config.device)
    else:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.expanduser(self.config.checkpoint_path)

    # If checkpoint_path is a directory with config.json or a Hugging Face repo ID:
    is_local_dir = os.path.isdir(checkpoint_path)
    is_local_file = os.path.isfile(checkpoint_path)

    if is_local_dir or not is_local_file:
      # Load via PyTorchModelHubMixin.from_pretrained (downloads config.json and weights)
      self.model = torch_model_lib.TimesFM3Torch.from_pretrained(
        checkpoint_path,
        cache_dir=self.config.cache_dir,
        force_download=self.config.force_download,
        token=self.config.token,
        revision=self.config.revision,
        local_files_only=self.config.local_files_only,
      )
      # Synchronize forecaster config with the loaded model config
      median_q_idx = self.config.median_quantile_index
      if median_q_idx >= len(self.model.quantiles):
        median_q_idx = len(self.model.quantiles) // 2

      self.config = dataclasses.replace(
        self.config,
        input_patch_length=self.model.input_patch_len,
        output_patch_length=self.model.output_patch_len,
        quantiles=list(self.model.quantiles),
        median_quantile_index=median_q_idx,
        residual_block_config=self.model.residual_block_config,
        transformer_config=self.model.transformer_config,
        use_variate_attention=self.model.use_variate_attention,
        value_clip=self.model.value_clip,
        use_stitching=self.model.use_stitching,
        use_linear_detrending=self.model.use_linear_detrending,
        linear_detrending_threshold=self.model.linear_detrending_threshold,
        use_iterative_cpm_revin=self.model.use_iterative_cpm_revin,
        use_frozen_running_stats=self.model.use_frozen_running_stats,
        input_transform=self.model.input_transform,
      )
    else:
      # Local file (.safetensors or .pth / .pt)
      self.model = _make_torch_model(self.config)
      if checkpoint_path.endswith(".safetensors"):
        state_dict = util.load_safetensors(checkpoint_path, device=self.device)
        self.model.load_state_dict(state_dict)
      elif checkpoint_path.endswith((".pth", ".pt")):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
      else:
        raise ValueError(
          f"Unsupported checkpoint path format: {checkpoint_path}. "
          "Expected .safetensors or .pth / .pt file."
        )

    self.model.to(self.device)
    self.model.eval()

  def predict(
    self,
    context: np.ndarray,
    horizon: int,
    past_only_covariates: np.ndarray | None = None,
    past_future_covariates: np.ndarray | None = None,
    ts_id: str | None = None,
    return_quantiles: bool = False,
    return_diagnostics: bool = False,
    use_symmetric_averaging: bool = False,
    make_positive: bool = False,
    sort_quantiles: bool = True,
    use_znorm: bool = False,
    padding_mode: str = "none",
  ) -> ForecastOutput:
    """Convenience wrapper: runs inference on a single time series."""
    results = list(
      self.predict_batch(
        contexts=[context],
        horizon=horizon,
        past_only_covariates=[past_only_covariates],
        past_future_covariates=[past_future_covariates],
        ts_ids=[ts_id] if ts_id is not None else None,
        return_quantiles=return_quantiles,
        return_diagnostics=return_diagnostics,
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
    return_diagnostics: bool = False,
    use_symmetric_averaging: bool = False,
    make_positive: bool = False,
    sort_quantiles: bool = True,
    use_znorm: bool = False,
    padding_mode: str = "none",
  ) -> Iterator[ForecastOutput]:
    """Runs inference on a batch of time series with optional covariates."""
    global_horizon = (
      math.ceil(horizon / self.config.output_patch_length)
      * self.config.output_patch_length
    )
    num_original_ts = len(contexts)
    original_ts_ids = list(ts_ids) if ts_ids is not None else [None] * num_original_ts

    if len(contexts) == 0:
      return

    po_cov_list = (
      list(past_only_covariates)
      if past_only_covariates is not None
      else [None] * num_original_ts
    )
    pf_cov_list = (
      list(past_future_covariates)
      if past_future_covariates is not None
      else [None] * num_original_ts
    )

    contexts_2d: list[np.ndarray] = []
    po_2d: list[np.ndarray | None] = []
    pf_2d: list[np.ndarray | None] = []

    for idx, ctx in enumerate(contexts):
      target_clean = np.atleast_2d(np.array(ctx, dtype=np.float32))
      po = po_cov_list[idx]
      po_arr = np.atleast_2d(np.array(po, dtype=np.float32)) if po is not None else None
      pf = pf_cov_list[idx]
      pf_arr = np.atleast_2d(np.array(pf, dtype=np.float32)) if pf is not None else None

      isnan = np.isnan(target_clean).all(axis=0)
      if isnan.all():
        first_valid_index = target_clean.shape[-1]
      else:
        first_valid_index = int(np.argmax(~isnan))

      if first_valid_index > 0 and first_valid_index < target_clean.shape[-1]:
        target_clean = target_clean[:, first_valid_index:]
        if po_arr is not None:
          po_arr = po_arr[:, first_valid_index:]
        if pf_arr is not None:
          pf_arr = pf_arr[:, first_valid_index:]
      elif first_valid_index == target_clean.shape[-1]:
        if target_clean.shape[-1] == 0:
          pass
        else:
          target_clean = np.zeros_like(target_clean)

      target_clean = linear_interpolation(target_clean)
      contexts_2d.append(target_clean)

      if po_arr is not None:
        po_arr = linear_interpolation(po_arr)
        po_2d.append(po_arr)
      else:
        po_2d.append(None)

      if pf_arr is not None:
        pf_arr = linear_interpolation(pf_arr)
        pf_2d.append(pf_arr)
      else:
        pf_2d.append(None)

    num_targets_in = contexts_2d[0].shape[0] if contexts_2d else 1

    for idx, ctx in enumerate(contexts_2d):
      if ctx.shape[0] != num_targets_in:
        raise ValueError(
          "All contexts must have the same number of target variates, but"
          f" contexts[0] has {num_targets_in} and contexts[{idx}] has"
          f" {ctx.shape[0]}."
        )

    was_1d_input = len(contexts) > 0 and np.ndim(contexts[0]) == 1

    znorm_per_example: list[list[tuple[float, float]]] = []
    if use_znorm:
      normed_contexts: list[np.ndarray] = []
      normed_po: list[np.ndarray | None] = []
      normed_pf: list[np.ndarray | None] = []

      for idx, ctx in enumerate(contexts_2d):
        stats_for_ex: list[tuple[float, float]] = []
        normed = ctx.copy()
        for r, row in enumerate(ctx):
          mu, sigma = _znorm_stats(row)
          stats_for_ex.append((mu, sigma))
          normed[r] = (row - mu) / sigma
        znorm_per_example.append(stats_for_ex)
        normed_contexts.append(normed)

        po = po_2d[idx]
        if po is not None:
          po_norm = po.copy()
          for r, row in enumerate(po):
            mu, sigma = _znorm_stats(row)
            po_norm[r] = (row - mu) / sigma
          normed_po.append(po_norm)
        else:
          normed_po.append(None)

        pf = pf_2d[idx]
        if pf is not None:
          pf_norm = pf.copy()
          for r, row in enumerate(pf):
            mu, sigma = _znorm_stats(row)
            pf_norm[r] = (row - mu) / sigma
          normed_pf.append(pf_norm)
        else:
          normed_pf.append(None)

      contexts_2d = normed_contexts
      po_2d = normed_po
      pf_2d = normed_pf
    else:
      for ctx in contexts_2d:
        znorm_per_example.append([(0.0, 1.0) for _ in range(ctx.shape[0])])

    if use_symmetric_averaging:
      sym_contexts: list[np.ndarray] = []
      sym_po: list[np.ndarray | None] = []
      sym_pf: list[np.ndarray | None] = []
      for idx, ctx in enumerate(contexts_2d):
        sym_contexts.append(ctx)
        sym_contexts.append(-ctx)
        po_val = po_2d[idx]
        sym_po.append(po_val)
        sym_po.append(None if po_val is None else -po_val)
        pf_val = pf_2d[idx]
        sym_pf.append(pf_val)
        sym_pf.append(None if pf_val is None else -pf_val)
      contexts_2d = sym_contexts
      po_2d = sym_po
      pf_2d = sym_pf

    if padding_mode == "edge":
      pad_len = global_horizon - horizon
      if pad_len > 0:
        for i, pf in enumerate(pf_2d):
          if pf is not None:
            pad_width = [(0, 0)] * (pf.ndim - 1) + [(0, pad_len)]
            pf_2d[i] = np.pad(pf, pad_width, mode="edge")
    elif padding_mode != "none":
      raise ValueError(f"Unknown padding_mode: '{padding_mode}'")

    queries: list[_Query] = []
    for idx, ctx in enumerate(contexts_2d):
      queries.append(
        _Query(
          horizon=global_horizon,
          targets=ctx,
          past_only_covariates=po_2d[idx],
          past_future_covariates=pf_2d[idx],
        )
      )

    if not queries:
      for i in range(num_original_ts):
        yield ForecastOutput(ts_id=original_ts_ids[i], forecast=None)
      return

    # Yield batches of size per_core_batch_size
    batch_size = self.config.per_core_batch_size
    ys = []
    num_queries = len(queries)
    num_batches = math.ceil(num_queries / batch_size)

    for i in range(num_batches):
      query_batch = queries[i * batch_size : (i + 1) * batch_size]
      if not query_batch:
        continue

      # Dynamic per-batch context length rounded up to patch length boundary
      max_ctx_in_batch = max(q.context_length for q in query_batch)
      batch_context = min(
        math.ceil(max_ctx_in_batch / self.config.input_patch_length)
        * self.config.input_patch_length,
        self.global_context,
      )
      batch_context = max(batch_context, self.config.input_patch_length)

      formatted_batch = [q.format(batch_context) for q in query_batch]

      (
        batched_hor,
        batched_tgt,
        batched_mask,
        batched_po,
        batched_pf,
      ) = tuple(list(w) for w in zip(*formatted_batch))

      tgt_torch = torch.from_numpy(np.stack(batched_tgt, axis=0)).to(
        self.device, dtype=torch.float32
      )
      mask_torch = torch.from_numpy(np.stack(batched_mask, axis=0)).to(
        self.device, dtype=torch.bool
      )

      po_torch = None
      if any(po is not None for po in batched_po):
        # A query without covariates in a batch where others have them gets
        # a zero placeholder shaped like the real covariate arrays (channel
        # count and width), not like the target: the covariate channel count
        # need not equal the number of target variates, and np.stack needs
        # every entry to have the same shape.
        po_ref = next(po for po in batched_po if po is not None)
        po_arrs = [po if po is not None else np.zeros_like(po_ref) for po in batched_po]
        po_torch = torch.from_numpy(np.stack(po_arrs, axis=0)).to(
          self.device, dtype=torch.float32
        )

      pf_torch = None
      if any(pf is not None for pf in batched_pf):
        # Same for past-future covariates. decode infers the horizon from
        # the covariate width (pf.shape[-1] - context) and slices the future
        # part, so the placeholder must match the real arrays' width as well
        # as their channel count. Real arrays in one batch already share both:
        # their context is the batch context and their future part is the
        # requested horizon.
        pf_ref = next(pf for pf in batched_pf if pf is not None)
        pf_arrs = [pf if pf is not None else np.zeros_like(pf_ref) for pf in batched_pf]
        pf_torch = torch.from_numpy(np.stack(pf_arrs, axis=0)).to(
          self.device, dtype=torch.float32
        )

      with torch.inference_mode():
        out_logits = self.model.decode(
          target=tgt_torch,
          horizon=batched_hor[0],
          past_only_covariates=po_torch,
          past_future_covariates=pf_torch,
          mask=mask_torch,
        )
      ys.append(out_logits.cpu().numpy())

    try_gc(self.device)
    all_raw_outputs = np.concatenate(ys, axis=0)
    all_raw_outputs = all_raw_outputs[:, :num_targets_in, :, :]

    if sort_quantiles:
      all_raw_outputs = np.sort(all_raw_outputs, axis=-1)

    if use_symmetric_averaging:
      ys_pos = all_raw_outputs[0::2]
      ys_neg = all_raw_outputs[1::2]
      if ys_pos.ndim >= 3 and ys_pos.shape[-1] > 1:
        all_raw_outputs = (ys_pos - ys_neg[..., ::-1]) / 2
      else:
        all_raw_outputs = (ys_pos - ys_neg) / 2

    if use_znorm:
      for i in range(all_raw_outputs.shape[0]):
        stats = znorm_per_example[i]
        for r in range(num_targets_in):
          mu, sigma = stats[r]
          all_raw_outputs[i, r] = all_raw_outputs[i, r] * sigma + mu

    if make_positive:
      for i in range(num_original_ts):
        if was_1d_input:
          if _is_nonnegative(contexts[i]):
            all_raw_outputs[i] = np.maximum(all_raw_outputs[i], 0.0)
        else:
          nonneg = _is_nonnegative(contexts[i])
          assert isinstance(nonneg, np.ndarray)
          for r in range(num_targets_in):
            if nonneg[r]:
              all_raw_outputs[i, r] = np.maximum(all_raw_outputs[i, r], 0.0)

    for i in range(num_original_ts):
      raw = all_raw_outputs[i]
      if was_1d_input:
        raw = raw[0]
        forecast = raw[:horizon, self.config.median_quantile_index]
        quantiles = raw[:horizon, :]
        yield ForecastOutput(
          ts_id=original_ts_ids[i],
          forecast=forecast,
          quantiles=quantiles if return_quantiles else None,
          diagnostics=(
            forecast_confidence_diagnostics(
              forecast, quantiles, self.config.quantiles
            )
            if return_diagnostics
            else None
          ),
        )
      else:
        forecast = raw[:, :horizon, self.config.median_quantile_index]
        quantiles = raw[:, :horizon, :]
        yield ForecastOutput(
          ts_id=original_ts_ids[i],
          forecast=forecast,
          quantiles=quantiles if return_quantiles else None,
          diagnostics=(
            forecast_confidence_diagnostics(
              forecast, quantiles, self.config.quantiles
            )
            if return_diagnostics
            else None
          ),
        )
