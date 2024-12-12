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
"""TimesFM JAX forecast API for inference."""

import logging
import multiprocessing
import time
from os import path
from typing import Any, Sequence

import einshape as es
import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download

from paxml import checkpoints, tasks_lib
from praxis import base_hyperparams, base_layer, pax_fiddle, py_utils, pytypes
from praxis.layers import normalizations, transformers
from timesfm import timesfm_base
from timesfm import patched_decoder

instantiate = base_hyperparams.instantiate
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor

_TOL = 1e-6


class TimesFmJax(timesfm_base.TimesFmBase):
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
  """

  def _get_sample_inputs(self):
    return {
        "input_ts":
            jnp.zeros(
                (
                    self.per_core_batch_size,
                    self.context_len + self.output_patch_len,
                ),
                dtype=jnp.float32,
            ),
        "input_padding":
            jnp.zeros(
                (
                    self.per_core_batch_size,
                    self.context_len + self.output_patch_len,
                ),
                dtype=jnp.float32,
            ),
        "freq":
            jnp.zeros(
                (
                    self.per_core_batch_size,
                    1,
                ),
                dtype=jnp.int32,
            ),
    }

  def __post_init__(self):
    self.num_cores = jax.local_device_count(self.backend)
    self.global_batch_size = self.per_core_batch_size * self.num_cores
    self._eval_context = base_layer.JaxContext.HParams(do_eval=True)
    self._pmapped_decode = None
    self._model = None
    self._train_state = None
    self._median_index = -1

  def load_from_checkpoint(
      self,
      checkpoint: timesfm_base.TimesFmCheckpoint,
  ) -> None:
    """Loads a checkpoint and compiles the decoder."""
    checkpoint_type = (checkpoints.CheckpointType.FLAX
                       if checkpoint.type is None else checkpoint.type)
    checkpoint_path = checkpoint.path
    step = checkpoint.step
    repo_id = checkpoint.huggingface_repo_id
    if checkpoint_path is None:
      checkpoint_path = path.join(snapshot_download(repo_id), "checkpoints")
    # Rewrite the devices for Jax.
    self.mesh_shape = [1, self.num_cores, 1]
    self.mesh_name = ["replica", "data", "mdl"]

    self.model_p = pax_fiddle.Config(
        patched_decoder.PatchedTimeSeriesDecoder,
        name="patched_decoder",
        horizon_len=self.output_patch_len,
        patch_len=self.input_patch_len,
        model_dims=self.model_dims,
        hidden_dims=self.model_dims,
        residual_block_tpl=pax_fiddle.Config(patched_decoder.ResidualBlock),
        quantiles=self.quantiles,
        use_freq=True,
        use_pos_emb=self.use_pos_emb,
        stacked_transformer_params_tpl=pax_fiddle.Config(
            transformers.StackedTransformer,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            transformer_layer_params_tpl=pax_fiddle.Config(
                transformers.Transformer,
                ln_tpl=pax_fiddle.Config(normalizations.RmsNorm,),
            ),
        ),
    )

    self._key1, self._key2 = jax.random.split(jax.random.PRNGKey(42))
    self._model = None
    self._train_state = None
    self._pmapped_decode = None
    self._eval_context = base_layer.JaxContext.HParams(do_eval=True)
    try:
      multiprocessing.set_start_method("spawn")
    except RuntimeError:
      print("Multiprocessing context has already been set.")
    # Download the checkpoint from Hugging Face Hub if not given

    #  Initialize the model weights.
    self._logging("Constructing model weights.")
    start_time = time.time()
    self._model = instantiate(self.model_p)
    var_weight_hparams = self._model.abstract_init_with_metadata(
        self._get_sample_inputs(), do_eval=True)
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
        f"Constructed model weights in {time.time() - start_time:.2f} seconds.")

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
        f"Restored checkpoint in {time.time() - start_time:.2f} seconds.")
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
        axis_size=self.num_cores,
    )
    with base_layer.JaxContext.new_context(hparams=self._eval_context):
      _ = self._pmapped_decode(
          NestedMap({
              "input_ts":
                  jnp.zeros(
                      (
                          self.num_cores,
                          self.per_core_batch_size,
                          self.context_len,
                      ),
                      dtype=jnp.float32,
                  ),
              "input_padding":
                  jnp.zeros(
                      (
                          self.num_cores,
                          self.per_core_batch_size,
                          self.context_len + self.horizon_len,
                      ),
                      dtype=jnp.float32,
                  ),
              "date_features":
                  None,
              "freq":
                  jnp.zeros(
                      (self.num_cores, self.per_core_batch_size, 1),
                      dtype=jnp.int32,
                  ),
          }))
    self._logging(f"Jitted decoding in {time.time() - start_time:.2f} seconds.")

  def _forecast(
      self,
      inputs: Sequence[Any],
      freq: Sequence[int] | None = None,
      window_size: int | None = None,
      forecast_context_len: int | None = None,
      return_forecast_on_context: bool = False,
  ) -> tuple[np.ndarray, np.ndarray]:
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
          " `forecast`.")
    if forecast_context_len is None:
      fcontext_len = self.context_len
    else:
      fcontext_len = forecast_context_len
    inputs = [np.array(ts)[-fcontext_len:] for ts in inputs]

    if window_size is not None:
      new_inputs = []
      for ts in inputs:
        new_inputs.extend(timesfm_base.moving_average(ts, window_size))
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
        input_ts_in = jnp.array(input_ts[i * self.global_batch_size:(i + 1) *
                                         self.global_batch_size])
        input_padding_in = jnp.array(
            input_padding[i * self.global_batch_size:(i + 1) *
                          self.global_batch_size],)
        inp_freq_in = jnp.array(
            inp_freq[i * self.global_batch_size:(i + 1) *
                     self.global_batch_size, :],
            dtype=jnp.int32,
        )
        pmapped_inputs = NestedMap({
            "input_ts":
                es.jax_einshape(
                    "(db)...->db...",
                    input_ts_in,
                    d=self.num_cores,
                ),
            "input_padding":
                es.jax_einshape(
                    "(db)...->db...",
                    input_padding_in,
                    d=self.num_cores,
                ),
            "date_features":
                None,
            "freq":
                es.jax_einshape(
                    "(db)...->db...",
                    inp_freq_in,
                    d=self.num_cores,
                ),
        })
        mean_output, full_output = self._pmapped_decode(pmapped_inputs)
        if not return_forecast_on_context:
          mean_output = mean_output[:, :, self._horizon_start:, ...]
          full_output = full_output[:, :, self._horizon_start:, ...]
        mean_output = es.jax_einshape("db...->(db)...",
                                      mean_output,
                                      d=self.num_cores)
        full_output = es.jax_einshape("db...->(db)...",
                                      full_output,
                                      d=self.num_cores)
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
    return mean_outputs, full_outputs
