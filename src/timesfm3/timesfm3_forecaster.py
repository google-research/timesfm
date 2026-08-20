"""Forecaster API wrapping a pretrained TimesFM3 PyTorch model."""

from __future__ import annotations

import dataclasses
import gc
import logging
import math
import os
from typing import Any, Iterator

import numpy as np
import torch

from . import configs
from . import model as torch_model_lib
from . import util

_MAX_CONTEXT_LENGTH = 16384
_SIGMA_THRESHOLD: float = 1e-7
_GC_MEMORY_THRESHOLD: float = 0.9


@dataclasses.dataclass
class _ModelConfig:
  """Configuration for a PyTorch TimesFM3 forecaster."""

  # Path to checkpoint file (.pth or .safetensors).
  checkpoint_path: str = "~/models/tfm3/timesfm3_torch_1944000.safetensors"

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


def strip_leading_nans(arr: np.ndarray) -> np.ndarray:
  """Removes contiguous NaN values from the beginning of a NumPy array."""
  if arr.size == 0:
    return arr

  was_1d = arr.ndim == 1
  arr2d = np.atleast_2d(arr)

  isnan = np.atleast_1d(np.isnan(arr2d).all(axis=0))
  first_valid_index = int(np.argmax(~isnan))

  if first_valid_index == 0 and isnan[0]:
    if was_1d:
      return np.array([], dtype=arr.dtype)
    return np.empty((arr2d.shape[0], 0), dtype=arr.dtype)

  result = arr2d[:, first_valid_index:]
  return result[0] if was_1d else result


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
      valid_indices = valid_mask.nonzero()[0]
      valid_values = row[valid_mask]
      row[nan_mask[r]] = np.interp(nan_indices, valid_indices, valid_values)

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
  """A single forecast query."""

  horizon: int
  targets: np.ndarray
  past_only_covariates: np.ndarray | None = None
  past_future_covariates: np.ndarray | None = None
  padded: bool = False

  @property
  def context_length(self) -> int:
    return self.targets.shape[-1]

  def format(self, global_context: int) -> tuple[
      int,
      np.ndarray,
      np.ndarray,
      np.ndarray | None,
      np.ndarray | None,
      bool,
  ]:
    """Formats and left-pads/truncates the query to global_context length."""
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

    if self.context_length > global_context:
      targets = targets[:, -global_context:]
      masks = masks[-global_context:]
      if past_only_covariates is not None:
        past_only_covariates = past_only_covariates[:, -global_context:]
      if past_future_covariates is not None:
        past_future_covariates = past_future_covariates[
            :, -(global_context + self.horizon) :
        ]
    elif self.context_length < global_context:
      pad_len = global_context - self.context_length
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
      if self.padded:
        masks = np.ones_like(masks, dtype=bool)
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
        self.padded,
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

    self.model = _make_torch_model(self.config)

    checkpoint_path = os.path.expanduser(self.config.checkpoint_path)
    if os.path.exists(checkpoint_path):
      if checkpoint_path.endswith(".safetensors"):
        state_dict = util.load_safetensors(checkpoint_path, device=self.device)
        self.model.load_state_dict(state_dict)
      elif checkpoint_path.endswith(".pth") or checkpoint_path.endswith(".pt"):
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
      use_symmetric_averaging: bool = False,
      make_positive: bool = False,
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
            use_symmetric_averaging=use_symmetric_averaging,
            make_positive=make_positive,
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
      use_znorm: bool = False,
      padding_mode: str = "none",
  ) -> Iterator[ForecastOutput]:
    """Runs inference on a batch of time series with optional covariates."""
    global_horizon = (
        math.ceil(horizon / self.config.output_patch_length)
        * self.config.output_patch_length
    )
    num_original_ts = len(contexts)
    original_ts_ids = (
        list(ts_ids) if ts_ids is not None else [None] * num_original_ts
    )

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
      arr = np.atleast_2d(np.array(ctx, dtype=np.float64))
      arr = strip_leading_nans(arr)
      arr = linear_interpolation(arr)
      contexts_2d.append(arr)

      po = po_cov_list[idx]
      if po is not None:
        po_arr = np.atleast_2d(np.array(po, dtype=np.float64))
        po_arr = strip_leading_nans(po_arr)
        po_arr = linear_interpolation(po_arr)
        po_2d.append(po_arr)
      else:
        po_2d.append(None)

      pf = pf_cov_list[idx]
      if pf is not None:
        pf_arr = np.atleast_2d(np.array(pf, dtype=np.float64))
        pf_arr = strip_leading_nans(pf_arr)
        pf_arr = linear_interpolation(pf_arr)
        pf_2d.append(pf_arr)
      else:
        pf_2d.append(None)

    original_contexts = list(contexts_2d)
    num_targets_in = contexts_2d[0].shape[0] if contexts_2d else 1

    for idx, ctx in enumerate(contexts_2d):
      if ctx.shape[0] != num_targets_in:
        raise ValueError(
            "All contexts must have the same number of target variates, but"
            f" contexts[0] has {num_targets_in} and contexts[{idx}] has"
            f" {ctx.shape[0]}."
        )

    is_univariate = num_targets_in == 1

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
      while len(query_batch) < batch_size:
        query_batch.append(dataclasses.replace(query_batch[0], padded=True))
      formatted_batch = [q.format(self.global_context) for q in query_batch]

      (
          batched_hor,
          batched_tgt,
          batched_mask,
          batched_po,
          batched_pf,
          _,
      ) = tuple(list(w) for w in zip(*formatted_batch))

      tgt_torch = torch.from_numpy(np.stack(batched_tgt, axis=0)).to(
          self.device, dtype=torch.float32
      )
      mask_torch = torch.from_numpy(np.stack(batched_mask, axis=0)).to(
          self.device, dtype=torch.bool
      )

      po_torch = None
      if any(po is not None for po in batched_po):
        po_arrs = [
            po if po is not None else np.zeros_like(batched_tgt[j])
            for j, po in enumerate(batched_po)
        ]
        po_torch = torch.from_numpy(np.stack(po_arrs, axis=0)).to(
            self.device, dtype=torch.float32
        )

      pf_torch = None
      if any(pf is not None for pf in batched_pf):
        pf_arrs = [
            pf if pf is not None else np.zeros_like(batched_tgt[j])
            for j, pf in enumerate(batched_pf)
        ]
        pf_torch = torch.from_numpy(np.stack(pf_arrs, axis=0)).to(
            self.device, dtype=torch.float32
        )

      with torch.no_grad():
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

    num_relevant_outputs = (
        2 * num_original_ts if use_symmetric_averaging else num_original_ts
    )
    all_raw_outputs = all_raw_outputs[
        :num_relevant_outputs, :num_targets_in, :, :
    ]

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
        if is_univariate:
          if _is_nonnegative(original_contexts[i]):
            all_raw_outputs[i] = np.maximum(all_raw_outputs[i], 0.0)
        else:
          nonneg = _is_nonnegative(original_contexts[i])
          assert isinstance(nonneg, np.ndarray)
          for r in range(num_targets_in):
            if nonneg[r]:
              all_raw_outputs[i, r] = np.maximum(all_raw_outputs[i, r], 0.0)

    for i in range(num_original_ts):
      raw = all_raw_outputs[i]
      if is_univariate:
        raw = raw[0]
        yield ForecastOutput(
            ts_id=original_ts_ids[i],
            forecast=raw[:horizon, self.config.median_quantile_index],
            quantiles=raw[:horizon, :] if return_quantiles else None,
        )
      else:
        yield ForecastOutput(
            ts_id=original_ts_ids[i],
            forecast=raw[:, :horizon, self.config.median_quantile_index],
            quantiles=raw[:, :horizon, :] if return_quantiles else None,
        )
