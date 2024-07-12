# Copyright 2024 Google LLC
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

"""TimesFM forecast API for inference."""

import collections
import logging
import multiprocessing
from os import path
import time
from typing import Any, Literal, Optional, Sequence

import einshape as es
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from paxml import checkpoints
from paxml import tasks_lib
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import normalizations
from praxis.layers import transformers

from utilsforecast.processing import make_future_dataframe

from . import patched_decoder
from . import xreg_lib

instantiate = base_hyperparams.instantiate
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
Category = xreg_lib.Category
XRegMode = xreg_lib.XRegMode

_TOL = 1e-6


def process_group(key, group, value_name, forecast_context_len):
  group = group.tail(forecast_context_len)
  return np.array(group[value_name], dtype=np.float32), key


def moving_average(arr, window_size):
  """Calculates the moving average using NumPy's convolution function."""
  # Pad with zeros to handle initial window positions
  arr_padded = np.pad(arr, (window_size - 1, 0), "constant")
  smoothed_arr = (
      np.convolve(arr_padded, np.ones(window_size), "valid") / window_size
  )
  return [smoothed_arr, arr - smoothed_arr]


def freq_map(freq: str):
  """Returns the frequency map for the given frequency string."""
  freq = str.upper(freq)
  if (
      freq.endswith("H")
      or freq.endswith("T")
      or freq.endswith("MIN")
      or freq.endswith("D")
      or freq.endswith("B")
      or freq.endswith("U")
  ):
    return 0
  elif freq.endswith(("W", "M", "MS")):
    return 1
  elif freq.endswith("Y") or freq.endswith("Q"):
    return 2
  else:
    raise ValueError(f"Invalid frequency: {freq}")


# Per time series normalization: forward.
def _normalize(batch):
  stats = [
      (np.mean(x), np.where((w := np.std(x)) > _TOL, w, 1.0)) for x in batch
  ]
  new_batch = [(x - stat[0]) / stat[1] for x, stat in zip(batch, stats)]
  return new_batch, stats


# Per time series normalization: inverse.
def _renormalize(batch, stats):
  return [x * stat[1] + stat[0] for x, stat in zip(batch, stats)]


class TimesFm:
  """TimesFM forecast API for inference.

  This class is the scaffolding for calling TimesFM forecast. To properly use:
    1. Create an instance with the correct hyperparameters of a TimesFM model.
    2. Call `load_from_checkpoint` to load a compatible checkpoint.
    3. Call `forecast` for inference.

  Given the model size, this API does not shard the model weights for SPMD. All
  parallelism happens on the data dimension.

  Compilation happens during the first time `forecast` is called and uses the
  `per_core_batch_size` to set and freeze the input signature. Subsequent calls
  to `forecast` reflect the actual inference latency.

  Attributes:
    per_core_batch_size: Batch size on each core for data parallelism.
    backend: One of "cpu", "gpu" or "tpu".
    num_devices: Number of cores provided the backend.
    global_batch_size: per_core_batch_size * num_devices. Each batch of
      inference task will be padded with respect to global_batch_size to
      minimize latency.
    context_len: Largest context length the model allows for each decode call.
      This technically can be any large, but practically should set to the
      context length the checkpoint was trained with.
    horizon_len: Forecast horizon.
    input_patch_len: Input patch len.
    output_patch_len: Output patch len. How many timepoints is taken from a
      single step of autoregressive decoding. Can be set as the training horizon
      of the checkpoint.
    mesh_shape: Shape of the data parallelism mesh.
    mesh_name: Names of the data parallelism mesh.
    model_p: Configuration of the TimesFM model deduced from the hparams.
  """

  def _logging(self, s):
    if self._verbose:
      print(s)

  def __init__(
      self,
      context_len: int,
      horizon_len: int,
      input_patch_len: int,
      output_patch_len: int,
      num_layers: int,
      model_dims: int,
      per_core_batch_size: int = 32,
      backend: Literal["cpu", "gpu", "tpu"] = "cpu",
      quantiles: Sequence[float] | None = None,
      verbose: bool = True,
  ) -> None:
    """Initializes the TimesFM forecast API.

    Args:
      context_len: Largest context length the model allows for each decode call.
        This technically can be any large, but practically should set to the
        context length the checkpoint was trained with.
      horizon_len: Forecast horizon.
      input_patch_len: Input patch len.
      output_patch_len: Output patch len. How many timepoints is taken from a
        single step of autoregressive decoding. Can be set as the training
        horizon of the checkpoint.
      num_layers: Number of transformer layers.
      model_dims: Model dimension.
      per_core_batch_size: Batch size on each core for data parallelism.
      backend: One of "cpu", "gpu" or "tpu".
      quantiles: list of output quantiles supported by the model.
      verbose: Whether to print logging messages.
    """
    self.per_core_batch_size = per_core_batch_size
    self.backend = backend
    self.num_devices = jax.local_device_count(self.backend)
    self.global_batch_size = self.per_core_batch_size * self.num_devices

    self.context_len = context_len
    self.horizon_len = horizon_len
    self.input_patch_len = input_patch_len
    self.output_patch_len = output_patch_len
    self._horizon_start = self.context_len - self.input_patch_len

    self.mesh_shape = [1, self.num_devices, 1]
    self.mesh_name = ["replica", "data", "mdl"]
    if quantiles is None:
      quantiles = patched_decoder.DEFAULT_QUANTILES

    self.model_p = pax_fiddle.Config(
        patched_decoder.PatchedTimeSeriesDecoder,
        name="patched_decoder",
        horizon_len=self.output_patch_len,
        patch_len=input_patch_len,
        model_dims=model_dims,
        hidden_dims=model_dims,
        residual_block_tpl=pax_fiddle.Config(patched_decoder.ResidualBlock),
        quantiles=quantiles,
        use_freq=True,
        stacked_transformer_params_tpl=pax_fiddle.Config(
            transformers.StackedTransformer,
            num_heads=16,
            num_layers=num_layers,
            transformer_layer_params_tpl=pax_fiddle.Config(
                transformers.Transformer,
                ln_tpl=pax_fiddle.Config(
                    normalizations.RmsNorm,
                ),
            ),
        ),
    )

    self._key1, self._key2 = jax.random.split(jax.random.PRNGKey(42))
    self._model = None
    self._train_state = None
    self._pmapped_decode = None
    self._verbose = verbose
    self._eval_context = base_layer.JaxContext.HParams(do_eval=True)
    try:
      multiprocessing.set_start_method("spawn")
    except RuntimeError:
      print("Multiprocessing context has already been set.")

  def _get_sample_inputs(self):
    return {
        "input_ts": jnp.zeros(
            (
                self.per_core_batch_size,
                self.context_len + self.output_patch_len,
            ),
            dtype=jnp.float32,
        ),
        "input_padding": jnp.zeros(
            (
                self.per_core_batch_size,
                self.context_len + self.output_patch_len,
            ),
            dtype=jnp.float32,
        ),
        "freq": jnp.zeros(
            (
                self.per_core_batch_size,
                1,
            ),
            dtype=jnp.int32,
        ),
    }

  def load_from_checkpoint(
      self,
      checkpoint_path: Optional[str] = None,
      repo_id: str = "google/timesfm-1.0-200m",
      checkpoint_type: checkpoints.CheckpointType = checkpoints.CheckpointType.FLAX,
      step: int | None = None,
  ) -> None:
    """Loads a checkpoint and compiles the decoder.

    Args:
      checkpoint_path: Optional path to the checkpoint directory.
      repo_id: Hugging Face Hub repo id.
      checkpoint_type: type of PAX checkpoint
      step: step of the checkpoint to load. If `None`, load latest checkpoint.
    """
    # Download the checkpoint from Hugging Face Hub if not given
    if checkpoint_path is None:
      checkpoint_path = path.join(snapshot_download(repo_id), "checkpoints")

    #  Initialize the model weights.
    self._logging("Constructing model weights.")
    start_time = time.time()
    self._model = instantiate(self.model_p)
    var_weight_hparams = self._model.abstract_init_with_metadata(
        self._get_sample_inputs(), do_eval=True
    )
    train_state_partition_specs = tasks_lib.create_state_partition_specs(
        var_weight_hparams,
        mesh_shape=self.mesh_shape,
        mesh_axis_names=self.mesh_name,
        discard_opt_states=True,
        learners=None,
    )
    train_state_local_shapes = tasks_lib.create_state_unpadded_shapes(
        var_weight_hparams,
        discard_opt_states=True,
        learners=None,
    )
    self._logging(
        f"Constructed model weights in {time.time() - start_time:.2f} seconds."
    )

    # Load the model weights.
    self._logging(f"Restoring checkpoint from {checkpoint_path}.")
    start_time = time.time()
    self._train_state = checkpoints.restore_checkpoint(
        train_state_local_shapes,
        checkpoint_dir=checkpoint_path,
        checkpoint_type=checkpoint_type,
        state_specs=train_state_partition_specs,
        step=step,
    )
    self._logging(
        f"Restored checkpoint in {time.time() - start_time:.2f} seconds."
    )
    self.jit_decode()

  def jit_decode(self):
    """Jitting decoding function."""

    # Initialize and jit the decode fn.
    def _decode(inputs):
      assert self._model is not None
      assert self._train_state is not None
      return self._model.apply(
          self._train_state.mdl_vars,
          inputs,
          horizon_len=self.horizon_len,
          output_patch_len=self.output_patch_len,
          max_len=self.context_len,
          return_forecast_on_context=True,
          rngs={
              base_layer.PARAMS: self._key1,
              base_layer.RANDOM: self._key2,
          },
          method=self._model.decode,
      )

    self._logging("Jitting decoding.")
    start_time = time.time()
    self._pmapped_decode = jax.pmap(
        _decode,
        axis_name="batch",
        devices=jax.devices(self.backend),
        backend=self.backend,
        axis_size=self.num_devices,
    )
    with base_layer.JaxContext.new_context(hparams=self._eval_context):
      _ = self._pmapped_decode(
          NestedMap({
              "input_ts": jnp.zeros(
                  (
                      self.num_devices,
                      self.per_core_batch_size,
                      self.context_len,
                  ),
                  dtype=jnp.float32,
              ),
              "input_padding": jnp.zeros(
                  (
                      self.num_devices,
                      self.per_core_batch_size,
                      self.context_len + self.horizon_len,
                  ),
                  dtype=jnp.float32,
              ),
              "date_features": None,
              "freq": jnp.zeros(
                  (self.num_devices, self.per_core_batch_size, 1),
                  dtype=jnp.int32,
              ),
          })
      )
    self._logging(f"Jitted decoding in {time.time() - start_time:.2f} seconds.")

  def _preprocess(
      self, inputs: Sequence[np.array], freq: Sequence[int]
  ) -> tuple[np.array, np.array, int]:
    """Formats and pads raw inputs to feed into the model.

    This function both pads each time series to match the context length, and
    pads the inputs to meet the SPMD shape requirement.

    Args:
      inputs: A list of 1d JTensors. Each JTensor is the context time series of
        a single forecast task.
      freq: list of frequencies

    Returns:
    A tuple of:
    - the padded input time series to meet the model required context.
    - the padding indicator.
    - the number of padded examples for SPMD so that each core has the same
        number (a multiple of `batch_size`) of examples.
    """

    input_ts, input_padding, inp_freq = [], [], []

    pmap_pad = (
        (len(inputs) - 1) // self.global_batch_size + 1
    ) * self.global_batch_size - len(inputs)

    for i, ts in enumerate(inputs):
      input_len = ts.shape[0]
      padding = np.zeros(shape=(input_len + self.horizon_len,), dtype=float)
      if input_len < self.context_len:
        num_front_pad = self.context_len - input_len
        ts = np.concatenate(
            [np.zeros(shape=(num_front_pad,), dtype=float), ts], axis=0
        )
        padding = np.concatenate(
            [np.ones(shape=(num_front_pad,), dtype=float), padding], axis=0
        )
      elif input_len > self.context_len:
        ts = ts[-self.context_len :]
        padding = padding[-(self.context_len + self.horizon_len) :]

      input_ts.append(ts)
      input_padding.append(padding)
      inp_freq.append(freq[i])

    # Padding the remainder batch.
    for _ in range(pmap_pad):
      input_ts.append(input_ts[-1])
      input_padding.append(input_padding[-1])
      inp_freq.append(inp_freq[-1])

    return (
        np.stack(input_ts, axis=0),
        np.stack(input_padding, axis=0),
        np.array(inp_freq).astype(np.int32).reshape(-1, 1),
        pmap_pad,
    )

  def forecast(
      self,
      inputs: Sequence[Any],
      freq: Sequence[int] | None = None,
      window_size: int | None = None,
      forecast_context_len: int | None = None,
      return_forecast_on_context: bool = False,
  ) -> tuple[JTensor, JTensor]:
    """Forecasts on a list of time series.

    Args:
      inputs: list of time series forecast contexts. Each context time series
        should be in a format convertible to JTensor by `jnp.array`.
      freq: frequency of each context time series. 0 for high frequency
        (default), 1 for medium, and 2 for low. Notice this is different from
        the `freq` required by `forecast_on_df`.
      window_size: window size of trend + residual decomposition. If None then
        we do not do decomposition.
      forecast_context_len: optional max context length.
      return_forecast_on_context: True to return the forecast on the context
        when available, i.e. after the first input patch.

    Returns:
    A tuple for JTensors:
    - the mean forecast of size (# inputs, # forecast horizon),
    - the full forecast (mean + quantiles) of size
        (# inputs,  # forecast horizon, 1 + # quantiles).

    Raises:
    ValueError: If the checkpoint is not properly loaded.
    """
    if not self._train_state or not self._model:
      raise ValueError(
          "Checkpoint not loaded. Call `load_from_checkpoint` before"
          " `forecast`."
      )
    if forecast_context_len is None:
      forecast_context_len = self.context_len
    inputs = [np.array(ts)[-forecast_context_len:] for ts in inputs]
    inp_min = np.min([np.min(ts) for ts in inputs])

    if window_size is not None:
      new_inputs = []
      for ts in inputs:
        new_inputs.extend(moving_average(ts, window_size))
      inputs = new_inputs

    if freq is None:
      logging.info("No frequency provided via `freq`. Default to high (0).")
      freq = [0] * len(inputs)

    input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(inputs, freq)
    with base_layer.JaxContext.new_context(hparams=self._eval_context):
      mean_outputs = []
      full_outputs = []
      assert input_ts.shape[0] % self.global_batch_size == 0
      for i in range(input_ts.shape[0] // self.global_batch_size):
        input_ts_in = jnp.array(
            input_ts[
                i * self.global_batch_size : (i + 1) * self.global_batch_size
            ]
        )
        input_padding_in = jnp.array(
            input_padding[
                i * self.global_batch_size : (i + 1) * self.global_batch_size
            ],
        )
        inp_freq_in = jnp.array(
            inp_freq[
                i * self.global_batch_size : (i + 1) * self.global_batch_size, :
            ],
            dtype=jnp.int32,
        )
        pmapped_inputs = NestedMap({
            "input_ts": es.jax_einshape(
                "(db)...->db...",
                input_ts_in,
                d=self.num_devices,
            ),
            "input_padding": es.jax_einshape(
                "(db)...->db...",
                input_padding_in,
                d=self.num_devices,
            ),
            "date_features": None,
            "freq": es.jax_einshape(
                "(db)...->db...",
                inp_freq_in,
                d=self.num_devices,
            ),
        })
        mean_output, full_output = self._pmapped_decode(pmapped_inputs)
        if not return_forecast_on_context:
          mean_output = mean_output[:, :, self._horizon_start :, ...]
          full_output = full_output[:, :, self._horizon_start :, ...]
        mean_output = es.jax_einshape(
            "db...->(db)...", mean_output, d=self.num_devices
        )
        full_output = es.jax_einshape(
            "db...->(db)...", full_output, d=self.num_devices
        )
        mean_output = np.array(mean_output)
        full_output = np.array(full_output)
        mean_outputs.append(mean_output)
        full_outputs.append(full_output)

    mean_outputs = np.concatenate(mean_outputs, axis=0)
    full_outputs = np.concatenate(full_outputs, axis=0)

    if pmap_pad > 0:
      mean_outputs = mean_outputs[:-pmap_pad, ...]
      full_outputs = full_outputs[:-pmap_pad, ...]

    if window_size is not None:
      mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
      full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
    if inp_min >= 0:
      mean_outputs = np.maximum(mean_outputs, 0.0)
      full_outputs = np.maximum(full_outputs, 0.0)
    return mean_outputs, full_outputs

  def forecast_with_covariates(
      self,
      inputs: list[Sequence[float]],
      dynamic_numerical_covariates: (
          dict[str, Sequence[Sequence[float]]] | None
      ) = None,
      dynamic_categorical_covariates: (
          dict[str, Sequence[Sequence[Category]]] | None
      ) = None,
      static_numerical_covariates: dict[str, Sequence[float]] | None = None,
      static_categorical_covariates: (
          dict[str, Sequence[Category]] | None
      ) = None,
      freq: Sequence[int] | None = None,
      window_size: int | None = None,
      forecast_context_len: int | None = None,
      xreg_mode: XRegMode = "xreg + timesfm",
      normalize_xreg_target_per_input: bool = True,
      ridge: float = 0.0,
      max_rows_per_col: int = 0,
      force_on_cpu: bool = False,
  ):
    """Forecasts on a list of time series with covariates.

    To optimize inference speed, avoid string valued categorical covariates.

    Args:
      inputs: A list of time series forecast contexts. Each context time series
        should be in a format convertible to JTensor by `jnp.array`.
      dynamic_numerical_covariates: A dict of dynamic numerical covariates.
      dynamic_categorical_covariates: A dict of dynamic categorical covariates.
      static_numerical_covariates: A dict of static numerical covariates.
      static_categorical_covariates: A dict of static categorical covariates.
      freq: frequency of each context time series. 0 for high frequency
        (default), 1 for medium, and 2 for low. Notice this is different from
        the `freq` required by `forecast_on_df`.
      window_size: window size of trend + residual decomposition. If None then
        we do not do decomposition.
      forecast_context_len: optional max context length.
      xreg_mode: one of "xreg + timesfm" or "timesfm + xreg". "xreg + timesfm"
        fits a model on the residuals of the TimesFM forecast. "timesfm + xreg"
        fits a model on the targets then forecasts on the residuals via TimesFM.
      normalize_xreg_target_per_input: whether to normalize the xreg target per
        input in the given batch.
      ridge: ridge penalty for the linear model.
      max_rows_per_col: max number of rows per column for the linear model.
      force_on_cpu: whether to force running on cpu for the linear model.

    Returns:
      A tuple of two lists. The first is the outputs of the model. The second is
      the outputs of the xreg.
    """

    # Verify and bookkeep covariates.
    if not (
        dynamic_numerical_covariates
        or dynamic_categorical_covariates
        or static_numerical_covariates
        or static_categorical_covariates
    ):
      raise ValueError(
          "At least one of dynamic_numerical_covariates,"
          " dynamic_categorical_covariates, static_numerical_covariates,"
          " static_categorical_covariates must be set."
      )

    # Track the lengths of (1) each input, (2) the part that can be used in the
    # linear model, and (3) the horizon.
    input_lens, train_lens, test_lens = [], [], []

    for i, input_ts in enumerate(inputs):
      input_len = len(input_ts)
      input_lens.append(input_len)

      if xreg_mode == "timesfm + xreg":
        # For fitting residuals, no TimesFM forecast on the first patch.
        train_lens.append(max(0, input_len - self.input_patch_len))
      elif xreg_mode == "xreg + timesfm":
        train_lens.append(input_len)
      else:
        raise ValueError(f"Unsupported mode: {xreg_mode}")

      if dynamic_numerical_covariates:
        test_lens.append(
            len(list(dynamic_numerical_covariates.values())[0][i]) - input_len
        )
      elif dynamic_categorical_covariates:
        test_lens.append(
            len(list(dynamic_categorical_covariates.values())[0][i]) - input_len
        )
      else:
        test_lens.append(self.horizon_len)

      if test_lens[-1] > self.horizon_len:
        raise ValueError(
            "Forecast requested longer horizon than the model definition "
            f"supports: {test_lens[-1]} vs {self.horizon_len}."
        )

    # Prepare the covariates into train and test.
    train_dynamic_numerical_covariates = collections.defaultdict(list)
    test_dynamic_numerical_covariates = collections.defaultdict(list)
    train_dynamic_categorical_covariates = collections.defaultdict(list)
    test_dynamic_categorical_covariates = collections.defaultdict(list)
    for covariates, train_covariates, test_covariates in (
        (
            dynamic_numerical_covariates,
            train_dynamic_numerical_covariates,
            test_dynamic_numerical_covariates,
        ),
        (
            dynamic_categorical_covariates,
            train_dynamic_categorical_covariates,
            test_dynamic_categorical_covariates,
        ),
    ):
      if not covariates:
        continue
      for covariate_name, covariate_values in covariates.items():
        for input_len, train_len, covariate_value in zip(
            input_lens, train_lens, covariate_values
        ):
          train_covariates[covariate_name].append(
              covariate_value[(input_len - train_len) : input_len]
          )
          test_covariates[covariate_name].append(covariate_value[input_len:])

    # Fit models.
    if xreg_mode == "timesfm + xreg":
      # Forecast via TimesFM then fit a model on the residuals.
      mean_outputs, _ = self.forecast(
          inputs,
          freq,
          window_size,
          forecast_context_len,
          return_forecast_on_context=True,
      )
      targets = [
          (
              np.array(input_ts)[-train_len:]
              - mean_output[
                  (self._horizon_start - train_len) : self._horizon_start
              ]
          )
          for input_ts, mean_output, train_len in zip(
              inputs, mean_outputs, train_lens
          )
      ]
      per_instance_stats = None
      if normalize_xreg_target_per_input:
        targets, per_instance_stats = _normalize(targets)
      xregs = xreg_lib.BatchedInContextXRegLinear(
          targets=targets,
          train_lens=train_lens,
          test_lens=test_lens,
          train_dynamic_numerical_covariates=train_dynamic_numerical_covariates,
          test_dynamic_numerical_covariates=test_dynamic_numerical_covariates,
          train_dynamic_categorical_covariates=train_dynamic_categorical_covariates,
          test_dynamic_categorical_covariates=test_dynamic_categorical_covariates,
          static_numerical_covariates=static_numerical_covariates,
          static_categorical_covariates=static_categorical_covariates,
      ).fit(
          ridge=ridge,
          one_hot_encoder_drop=None if ridge > 0 else "first",
          max_rows_per_col=max_rows_per_col,
          force_on_cpu=force_on_cpu,
          debug_info=False,
          assert_covariates=True,
          assert_covariate_shapes=True,
      )
      if normalize_xreg_target_per_input:
        xregs = _renormalize(xregs, per_instance_stats)
      outputs = [
          (
              mean_output[
                  self._horizon_start : (self._horizon_start + test_len)
              ]
              + xreg
          )
          for mean_output, test_len, xreg in zip(mean_outputs, test_lens, xregs)
      ]

    else:
      # Fit a model on the targets then forecast on the residuals via TimesFM.
      targets = [
          np.array(input_ts)[-train_len:]
          for input_ts, train_len in zip(inputs, train_lens)
      ]
      per_instance_stats = None
      if normalize_xreg_target_per_input:
        targets, per_instance_stats = _normalize(targets)
      xregs, xregs_on_context, _, _, _ = xreg_lib.BatchedInContextXRegLinear(
          targets=targets,
          train_lens=train_lens,
          test_lens=test_lens,
          train_dynamic_numerical_covariates=train_dynamic_numerical_covariates,
          test_dynamic_numerical_covariates=test_dynamic_numerical_covariates,
          train_dynamic_categorical_covariates=train_dynamic_categorical_covariates,
          test_dynamic_categorical_covariates=test_dynamic_categorical_covariates,
          static_numerical_covariates=static_numerical_covariates,
          static_categorical_covariates=static_categorical_covariates,
      ).fit(
          ridge=ridge,
          one_hot_encoder_drop=None if ridge > 0 else "first",
          max_rows_per_col=max_rows_per_col,
          force_on_cpu=force_on_cpu,
          debug_info=True,
          assert_covariates=True,
          assert_covariate_shapes=True,
      )
      mean_outputs, _ = self.forecast(
          [
              target - xreg_on_context
              for target, xreg_on_context in zip(targets, xregs_on_context)
          ],
          freq,
          window_size,
          forecast_context_len,
          return_forecast_on_context=True,
      )
      outputs = [
          (
              mean_output[
                  self._horizon_start : (self._horizon_start + test_len)
              ]
              + xreg
          )
          for mean_output, test_len, xreg in zip(mean_outputs, test_lens, xregs)
      ]
      if normalize_xreg_target_per_input:
        outputs = _renormalize(outputs, per_instance_stats)

    return outputs, xregs

  def forecast_on_df(
      self,
      inputs: pd.DataFrame,
      freq: str,
      forecast_context_len: int = 0,
      value_name: str = "values",
      model_name: str = "timesfm",
      window_size: int | None = None,
      num_jobs: int = 1,
  ) -> pd.DataFrame:
    """Forecasts on a list of time series.

    Args:
      inputs: A pd.DataFrame of all time series. The dataframe should have a
        `unique_id` column for identifying the time series, a `ds` column for
        timestamps and a value column for the time series values.
      freq: string valued `freq` of data. Notice this is different from the
        `freq` required by `forecast`. See `freq_map` for allowed values.
      forecast_context_len: If provided none zero, we take the last
        `forecast_context_len` time-points from each series as the forecast
        context instead of the `context_len` set by the model.
      value_name: The name of the value column.
      model_name: name of the model to be written into future df.
      window_size: window size of trend + residual decomposition. If None then
        we do not do decomposition.
      num_jobs: number of parallel processes to use for dataframe processing.

    Returns:
      Future forecasts dataframe.
    """
    if not (
        "unique_id" in inputs.columns
        and "ds" in inputs.columns
        and value_name in inputs.columns
    ):
      raise ValueError(
          f"DataFrame must have unique_id, ds and {value_name} columns."
      )
    if not forecast_context_len:
      forecast_context_len = self.context_len
    logging.info("Preprocessing dataframe.")
    df_sorted = inputs.sort_values(by=["unique_id", "ds"])
    new_inputs = []
    uids = []
    if num_jobs == 1:
      print("Processing dataframe with single process.")
      for key, group in df_sorted.groupby("unique_id"):
        inp, uid = process_group(
            key,
            group,
            value_name,
            forecast_context_len,
        )
        new_inputs.append(inp)
        uids.append(uid)
    else:
      if num_jobs == -1:
        num_jobs = multiprocessing.cpu_count()
      print("Processing dataframe with multiple processes.")
      with multiprocessing.Pool(processes=num_jobs) as pool:
        results = pool.starmap(
            process_group,
            [
                (key, group, value_name, forecast_context_len)
                for key, group in df_sorted.groupby("unique_id")
            ],
        )
      new_inputs, uids = zip(*results)
    print("Finished preprocessing dataframe.")
    freq_inps = [freq_map(freq)] * len(new_inputs)
    _, full_forecast = self.forecast(
        new_inputs, freq=freq_inps, window_size=window_size
    )
    print("Finished forecasting.")
    fcst_df = make_future_dataframe(
        uids=uids,
        last_times=df_sorted.groupby("unique_id")["ds"].tail(1),
        h=self.horizon_len,
        freq=freq,
    )
    fcst_df[model_name] = full_forecast[:, 0 : self.horizon_len, 0].reshape(
        -1, 1
    )

    if self._model.quantiles is not None:
      for i, q in enumerate(self._model.quantiles):
        q_col = f"{model_name}-q-{q}"
        fcst_df[q_col] = full_forecast[:, 0 : self.horizon_len, 1 + i].reshape(
            -1, 1
        )
        if q == 0.5:
          fcst_df[model_name] = fcst_df[q_col]
    logging.info("Finished creating output dataframe.")
    return fcst_df
