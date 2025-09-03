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
"""TimesFM pytorch forecast API for inference."""

import logging
from os import path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from timesfm import timesfm_base

from . import pytorch_patched_decoder as ppd

_TOL = 1e-6


class TimesFmTorch(timesfm_base.TimesFmBase):
  """TimesFM forecast API for inference."""

  def __post_init__(self):
    self._model_config = ppd.TimesFMConfig(
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        hidden_size=self.model_dims,
        intermediate_size=self.model_dims,
        patch_len=self.input_patch_len,
        horizon_len=self.output_patch_len,
        head_dim=self.model_dims // self.num_heads,
        quantiles=self.quantiles,
        use_positional_embedding=self.use_pos_emb,
    )
    self._model = None
    self.num_cores = 1
    self.global_batch_size = self.per_core_batch_size
    self._device = torch.device("cuda:0" if (
        torch.cuda.is_available() and self.backend == "gpu") else "cpu")
    self._median_index = -1

  def load_from_checkpoint(
      self,
      checkpoint: timesfm_base.TimesFmCheckpoint,
  ) -> None:
    """Loads a checkpoint and compiles the decoder."""
    checkpoint_path = checkpoint.path
    repo_id = checkpoint.huggingface_repo_id
    if checkpoint_path is None:
      checkpoint_path = path.join(
          snapshot_download(repo_id, local_dir=checkpoint.local_dir),
          "torch_model.ckpt")
    self._model = ppd.PatchedTimeSeriesDecoder(self._model_config)
    loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
    logging.info("Loading checkpoint from %s", checkpoint_path)
    self._model.load_state_dict(loaded_checkpoint)
    logging.info("Sending checkpoint to device %s", f"{self._device}")
    self._model.to(self._device)
    self._model.eval()
    # TODO: add compilation.

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
    if self._model is None:
      raise ValueError("Checkpoint is not properly loaded.")

    if forecast_context_len is None:
      forecast_context_len = self.context_len
    inputs = [np.array(ts)[-forecast_context_len:] for ts in inputs]

    if window_size is not None:
      new_inputs = []
      for ts in inputs:
        new_inputs.extend(timesfm_base.moving_average(ts, window_size))
      inputs = new_inputs

    if freq is None:
      logging.info("No frequency provided via `freq`. Default to high (0).")
      freq = [0] * len(inputs)

    input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(inputs, freq)

    with torch.no_grad():
      mean_outputs = []
      full_outputs = []
      for i in range(input_ts.shape[0] // self.global_batch_size):
        t_input_ts = torch.Tensor(input_ts[i * self.global_batch_size:(i + 1) *
                                           self.global_batch_size]).to(
                                               self._device)
        t_input_padding = torch.Tensor(
            input_padding[i * self.global_batch_size:(i + 1) *
                          self.global_batch_size]).to(self._device)
        t_inp_freq = torch.LongTensor(
            inp_freq[i * self.global_batch_size:(i + 1) *
                     self.global_batch_size, :]).to(self._device)

        mean_output, full_output = self._model.decode(
            input_ts=t_input_ts,
            paddings=t_input_padding,
            freq=t_inp_freq,
            horizon_len=self.horizon_len,
            output_patch_len=self.output_patch_len,
            # Returns forecasts on context for parity with the Jax version.
            return_forecast_on_context=True,
        )
        if not return_forecast_on_context:
          mean_output = mean_output[:, self._horizon_start:, ...]
          full_output = full_output[:, self._horizon_start:, ...]

        if self.backend == "gpu":
          mean_output = mean_output.cpu()
          full_output = full_output.cpu()
        mean_output = mean_output.detach().numpy()
        full_output = full_output.detach().numpy()
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
