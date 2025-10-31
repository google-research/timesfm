# Copyright 2025 Google LLC
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
"""TimesFM models."""

import dataclasses
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from huggingface_hub import ModelHubMixin, hf_hub_download
from safetensors.torch import load_file, save_file
from torch import nn

from .. import configs
from ..torch import dense, transformer, util
from . import timesfm_2p5_base

revin = util.revin


class TimesFM_2p5_200M_torch_module(nn.Module):
  """TimesFM 2.5 with 200M parameters."""

  config = timesfm_2p5_base.TimesFM_2p5_200M_Definition()

  def __init__(self):
    super().__init__()

    # Names constants.
    self.p = self.config.input_patch_len  # 32
    self.o = self.config.output_patch_len  # 128
    self.os = self.config.output_quantile_len  # 1024
    self.m = self.o // self.p  # 4
    self.x = self.config.stacked_transformers.num_layers  # 20
    self.h = self.config.stacked_transformers.transformer.num_heads  # 16
    self.md = self.config.stacked_transformers.transformer.model_dims  # 1280
    self.hd = self.md // self.h  # 80
    self.q = len(self.config.quantiles) + 1  # 10
    self.aridx = self.config.decode_index  # 5

    # Layers.
    self.tokenizer = dense.ResidualBlock(self.config.tokenizer)
    self.stacked_xf = nn.ModuleList(
      [
        transformer.Transformer(self.config.stacked_transformers.transformer)
        for _ in range(self.x)
      ]
    )
    self.output_projection_point = dense.ResidualBlock(
      self.config.output_projection_point
    )
    self.output_projection_quantiles = dense.ResidualBlock(
      self.config.output_projection_quantiles
    )

    # Device.
    if torch.cuda.is_available():
      self.device = torch.device("cuda:0")
      self.device_count = torch.cuda.device_count()
    else:
      self.device = torch.device("cpu")
      self.device_count = 1

  def load_checkpoint(self, path: str, **kwargs):
    """Loads a PyTorch TimesFM model from a checkpoint."""
    tensors = load_file(path)
    self.load_state_dict(tensors, strict=True)
    self.to(self.device)
    torch_compile = True
    if "torch_compile" in kwargs:
      torch_compile = kwargs["torch_compile"]
    if torch_compile:
      print("Compiling model...")
      self = torch.compile(self)

    self.eval()

  def forward(
    self,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    decode_caches: list[util.DecodeCache] | None = None,
  ):
    tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
    input_embeddings = self.tokenizer(tokenizer_inputs)

    if decode_caches is None:
      decode_caches = [None] * self.x

    output_embeddings = input_embeddings
    new_decode_caches = []
    for i, layer in enumerate(self.stacked_xf):
      output_embeddings, new_cache = layer(
        output_embeddings, masks[..., -1], decode_caches[i]
      )
      new_decode_caches.append(new_cache)
    output_ts = self.output_projection_point(output_embeddings)
    output_quantile_spread = self.output_projection_quantiles(output_embeddings)

    return (
      input_embeddings,
      output_embeddings,
      output_ts,
      output_quantile_spread,
    ), new_decode_caches

  def decode(self, horizon: int, inputs, masks):
    """Decodes the time series."""

    with torch.no_grad():
      batch_size, context = inputs.shape[0], inputs.shape[1]
      num_decode_steps = (horizon - 1) // self.o
      num_input_patches = context // self.p
      decode_cache_size = num_input_patches + num_decode_steps * self.m

      # Prefill
      patched_inputs = torch.reshape(inputs, (batch_size, -1, self.p))
      patched_masks = torch.reshape(masks, (batch_size, -1, self.p))

      # running stats
      n = torch.zeros(batch_size, device=inputs.device)
      mu = torch.zeros(batch_size, device=inputs.device)
      sigma = torch.zeros(batch_size, device=inputs.device)
      patch_mu = []
      patch_sigma = []
      for i in range(num_input_patches):
        (n, mu, sigma), _ = util.update_running_stats(
          n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)
      last_n, last_mu, last_sigma = n, mu, sigma
      context_mu = torch.stack(patch_mu, dim=1)
      context_sigma = torch.stack(patch_sigma, dim=1)

      decode_caches = [
        util.DecodeCache(
          next_index=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
          num_masked=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
          key=torch.zeros(
            batch_size,
            decode_cache_size,
            self.h,
            self.hd,
            device=inputs.device,
          ),
          value=torch.zeros(
            batch_size,
            decode_cache_size,
            self.h,
            self.hd,
            device=inputs.device,
          ),
        )
        for _ in range(self.x)
      ]

      normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
      normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
      (_, _, normed_outputs, normed_quantile_spread), decode_caches = self(
        normed_inputs, patched_masks, decode_caches
      )
      renormed_outputs = torch.reshape(
        revin(normed_outputs, context_mu, context_sigma, reverse=True),
        (batch_size, -1, self.o, self.q),
      )
      renormed_quantile_spread = torch.reshape(
        revin(normed_quantile_spread, context_mu, context_sigma, reverse=True),
        (batch_size, -1, self.os, self.q),
      )[:, -1, ...]

      # Autogressive decode
      ar_outputs = []
      last_renormed_output = renormed_outputs[:, -1, :, self.aridx]

      for _ in range(num_decode_steps):
        new_patched_input = torch.reshape(
          last_renormed_output, (batch_size, self.m, self.p)
        )
        new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)

        n, mu, sigma = last_n, last_mu, last_sigma
        new_mus, new_sigmas = [], []
        for i in range(self.m):
          (n, mu, sigma), _ = util.update_running_stats(
            n, mu, sigma, new_patched_input[:, i], new_mask[:, i]
          )
          new_mus.append(mu)
          new_sigmas.append(sigma)
        last_n, last_mu, last_sigma = n, mu, sigma
        new_mu = torch.stack(new_mus, dim=1)
        new_sigma = torch.stack(new_sigmas, dim=1)

        new_normed_input = revin(new_patched_input, new_mu, new_sigma, reverse=False)
        (_, _, new_normed_output, _), decode_caches = self(
          new_normed_input, new_mask, decode_caches
        )

        new_renormed_output = torch.reshape(
          revin(new_normed_output, new_mu, new_sigma, reverse=True),
          (batch_size, self.m, self.o, self.q),
        )
        ar_outputs.append(new_renormed_output[:, -1, ...])
        last_renormed_output = new_renormed_output[:, -1, :, self.aridx]

      if num_decode_steps > 0:
        ar_renormed_outputs = torch.stack(ar_outputs, dim=1)
      else:
        ar_renormed_outputs = None

    return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs

  def forecast_naive(
    self, horizon: int, inputs: Sequence[np.ndarray]
  ) -> list[np.ndarray]:
    """Forecasts the time series.

    This is a naive implementation for debugging purposes. No forecasting
    flags are used here. Forecasting quality can be subpar.

    Args:
      horizon: The number of time points to forecast.
      inputs: A sequence of numpy arrays, each representing a time series to
        query forecast for.

    Returns:
      A list of numpy arrays of forecasts.
    """
    outputs = []
    for each_input in inputs:
      input_t = torch.tensor(each_input, dtype=torch.float32)
      mask = torch.zeros_like(input_t, dtype=torch.bool)
      len_front_mask = self.p - (len(each_input) % self.p)
      if len_front_mask < self.p:
        input_t = torch.cat(
          [torch.zeros(len_front_mask, dtype=torch.float32), input_t], dim=0
        )
        mask = torch.cat([torch.ones(len_front_mask, dtype=torch.bool), mask], dim=0)
      input_t = input_t[None, ...]
      mask = mask[None, ...]
      t_pf, _, t_ar = self.decode(horizon, input_t, mask)
      to_concat = [t_pf[:, -1, ...]]
      if t_ar is not None:
        to_concat.append(t_ar.reshape(1, -1, self.q))
      torch_forecast = torch.cat(to_concat, dim=1)[..., :horizon]
      torch_forecast = torch_forecast.squeeze(0)
      outputs.append(torch_forecast.detach().cpu().numpy())
    return outputs


class TimesFM_2p5_200M_torch(timesfm_2p5_base.TimesFM_2p5, ModelHubMixin):
  """PyTorch implementation of TimesFM 2.5 with 200M parameters."""

  model: nn.Module = TimesFM_2p5_200M_torch_module()

  @classmethod
  def _from_pretrained(
    cls,
    *,
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    revision: Optional[str],
    cache_dir: Optional[Union[str, Path]],
    force_download: bool = True,
    proxies: Optional[Dict] = None,
    resume_download: Optional[bool] = None,
    local_files_only: bool,
    token: Optional[Union[str, bool]],
    **model_kwargs,
  ):
    """
    Loads a PyTorch safetensors TimesFM model from a local path or the Hugging
    Face Hub. This method is the backend for the `from_pretrained` class
    method provided by `ModelHubMixin`.
    """
    # Create an instance of the model wrapper class.
    instance = cls(**model_kwargs)
    # Download the config file for hf tracking.
    _ = hf_hub_download(
      repo_id="google/timesfm-2.5-200m-pytorch",
      filename="config.json",
      force_download=True,
    )
    print("Downloaded.")

    # Determine the path to the model weights.
    model_file_path = ""
    if os.path.isdir(model_id):
      logging.info("Loading checkpoint from local directory: %s", model_id)
      model_file_path = os.path.join(model_id, "model.safetensors")
      if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"model.safetensors not found in directory {model_id}")
    else:
      logging.info("Downloading checkpoint from Hugging Face repo %s", model_id)
      model_file_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors",
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        token=token,
        local_files_only=local_files_only,
      )

    logging.info("Loading checkpoint from: %s", model_file_path)
    # Load the weights into the model.
    instance.model.load_checkpoint(model_file_path, **model_kwargs)
    return instance

  def _save_pretrained(self, save_directory: Union[str, Path]):
    """
    Saves the model's state dictionary to a safetensors file. This method
    is called by the `save_pretrained` method from `ModelHubMixin`.
    """
    if not os.path.exists(save_directory):
      os.makedirs(save_directory)

    weights_path = os.path.join(save_directory, "model.safetensors")
    save_file(self.model.state_dict(), weights_path)

  def compile(self, forecast_config: configs.ForecastConfig, **kwargs) -> None:
    """Attempts to compile the model for fast decoding.

    See configs.ForecastConfig for more details on the supported flags.

    Args:
      forecast_config: Configuration for forecasting flags.
      **kwargs: Additional keyword arguments to pass to model.compile().
    """
    self.global_batch_size = (
      forecast_config.per_core_batch_size * self.model.device_count
    )

    # Shortcut.
    fc = forecast_config

    if fc.max_context % self.model.p != 0:
      logging.info(
        "When compiling, max context needs to be multiple of the patch size"
        " %d. Using max context = %d instead.",
        self.model.p,
        new_context := math.ceil(fc.max_context / self.model.p) * self.model.p,
      )
      fc = dataclasses.replace(fc, max_context=new_context)
    if fc.max_horizon % self.model.o != 0:
      logging.info(
        "When compiling, max horizon needs to be multiple of the output patch"
        " size %d. Using max horizon = %d instead.",
        self.model.o,
        new_horizon := math.ceil(fc.max_horizon / self.model.o) * self.model.o,
      )
      fc = dataclasses.replace(fc, max_horizon=new_horizon)
    if fc.max_context + fc.max_horizon > self.model.config.context_limit:
      raise ValueError(
        "Context + horizon must be less than the context limit."
        f" {fc.max_context} + {fc.max_horizon} >"
        f" {self.model.config.context_limit}."
      )
    if fc.use_continuous_quantile_head and (fc.max_horizon > self.model.os):
      raise ValueError(
        f"Continuous quantile head is not supported for horizons > {self.model.os}."
      )
    self.forecast_config = fc

    def _compiled_decode(horizon, inputs, masks):
      if horizon > fc.max_horizon:
        raise ValueError(
          f"Horizon must be less than the max horizon. {horizon} > {fc.max_horizon}."
        )

      inputs = (
        torch.from_numpy(np.array(inputs)).to(self.model.device).to(torch.float32)
      )
      masks = torch.from_numpy(np.array(masks)).to(self.model.device).to(torch.bool)
      batch_size = inputs.shape[0]

      if fc.infer_is_positive:
        is_positive = torch.all(inputs >= 0, dim=-1, keepdim=True)
      else:
        is_positive = None

      if fc.normalize_inputs:
        mu = torch.mean(inputs, dim=-1, keepdim=True)
        sigma = torch.std(inputs, dim=-1, keepdim=True)
        inputs = revin(inputs, mu, sigma, reverse=False)
      else:
        mu, sigma = None, None

      pf_outputs, quantile_spreads, ar_outputs = self.model.decode(
        forecast_config.max_horizon, inputs, masks
      )
      to_cat = [pf_outputs[:, -1, ...]]
      if ar_outputs is not None:
        to_cat.append(ar_outputs.reshape(batch_size, -1, self.model.q))
      full_forecast = torch.cat(to_cat, dim=1)

      def flip_quantile_fn(x):
        return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)

      if fc.force_flip_invariance:
        flipped_pf_outputs, flipped_quantile_spreads, flipped_ar_outputs = (
          self.model.decode(forecast_config.max_horizon, -inputs, masks)
        )
        flipped_quantile_spreads = flip_quantile_fn(flipped_quantile_spreads)
        flipped_pf_outputs = flip_quantile_fn(flipped_pf_outputs)
        to_cat = [flipped_pf_outputs[:, -1, ...]]
        if flipped_ar_outputs is not None:
          to_cat.append(flipped_ar_outputs.reshape(batch_size, -1, self.model.q))
        flipped_full_forecast = torch.cat(to_cat, dim=1)
        quantile_spreads = (quantile_spreads - flipped_quantile_spreads) / 2
        pf_outputs = (pf_outputs - flipped_pf_outputs) / 2
        full_forecast = (full_forecast - flipped_full_forecast) / 2

      if fc.use_continuous_quantile_head:
        for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
          full_forecast[:, :, quantile_index] = (
            quantile_spreads[:, : fc.max_horizon, quantile_index]
            - quantile_spreads[:, : fc.max_horizon, 5]
            + full_forecast[:, : fc.max_horizon, 5]
          )
      full_forecast = full_forecast[:, :horizon, :]

      if fc.return_backcast:
        full_backcast = pf_outputs[:, :-1, : self.model.p, :].reshape(
          batch_size, -1, self.model.q
        )
        full_forecast = torch.cat([full_backcast, full_forecast], dim=1)

      if fc.fix_quantile_crossing:
        for i in [4, 3, 2, 1]:
          full_forecast[:, :, i] = torch.where(
            full_forecast[:, :, i] < full_forecast[:, :, i + 1],
            full_forecast[:, :, i],
            full_forecast[:, :, i + 1],
          )
        for i in [6, 7, 8, 9]:
          full_forecast[:, :, i] = torch.where(
            full_forecast[:, :, i] > full_forecast[:, :, i - 1],
            full_forecast[:, :, i],
            full_forecast[:, :, i - 1],
          )

      if fc.normalize_inputs:
        full_forecast = revin(full_forecast, mu, sigma, reverse=True)

      if is_positive is not None:
        full_forecast = torch.where(
          is_positive[..., None],
          torch.maximum(full_forecast, torch.zeros_like(full_forecast)),
          full_forecast,
        )

      full_forecast = full_forecast.detach().cpu().numpy()
      return full_forecast[..., 5], full_forecast

    self.compiled_decode = _compiled_decode
