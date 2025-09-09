"""TimesFM model for TMax in Keras."""

import math
import tensorflow as tf
from google3.learning.timeseries.scalar.timesfm.tmax import configs
from google3.learning.timeseries.scalar.timesfm.tmax.keras import dense
from google3.learning.timeseries.scalar.timesfm.tmax.keras import transformer
from google3.learning.timeseries.scalar.timesfm.tmax.keras import util
from google3.learning.timeseries.scalar.timesfm.tmax.timesfm_2p5 import timesfm_2p5_base

revin = util.revin


class TimesFM_2p5_200M_keras_module(tf.keras.Model):
  """TimesFM 2.5 with 200M parameters."""

  config = timesfm_2p5_base.TimesFM_2p5_200M_Definition()

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Shorcuts.
    self.p = self.config.input_patch_len
    self.o = self.config.output_patch_len
    self.os = self.config.output_quantile_len
    self.m = self.o // self.p
    self.x = self.config.stacked_transformers.num_layers
    self.h = self.config.stacked_transformers.transformer.num_heads
    self.md = self.config.stacked_transformers.transformer.model_dims
    self.hd = self.md // self.h
    self.q = len(self.config.quantiles) + 1
    self.aridx = self.config.decode_index

    # Layers.
    self.tokenizer = dense.ResidualBlock(self.config.tokenizer)
    self.stacked_xf = [
        transformer.Transformer(self.config.stacked_transformers.transformer)
        for _ in range(self.x)
    ]
    self.output_projection_point = dense.ResidualBlock(
        self.config.output_projection_point
    )
    self.output_projection_quantiles = dense.ResidualBlock(
        self.config.output_projection_quantiles
    )

  def call(
      self,
      inputs: tf.Tensor,
      masks: tf.Tensor,
      decode_caches: list[dict] | None = None,
  ):
    tokenizer_inputs = tf.concat(
        [inputs, tf.cast(masks, inputs.dtype)], axis=-1
    )
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
    batch_size, context = tf.shape(inputs)[0], tf.shape(inputs)[1]
    num_decode_steps = (horizon - 1) // self.o
    num_input_patches = context // self.p
    decode_cache_size = num_input_patches + num_decode_steps * self.m

    # Prefill
    patched_inputs = tf.reshape(inputs, (batch_size, -1, self.p))
    patched_masks = tf.reshape(masks, (batch_size, -1, self.p))

    # running stats
    n = tf.zeros(batch_size, dtype=inputs.dtype)
    mu = tf.zeros(batch_size, dtype=inputs.dtype)
    sigma = tf.zeros(batch_size, dtype=inputs.dtype)
    patch_mu = []
    patch_sigma = []
    for i in range(num_input_patches):
      (n, mu, sigma), _ = util.update_running_stats(
          n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
      )
      patch_mu.append(mu)
      patch_sigma.append(sigma)
    last_n, last_mu, last_sigma = n, mu, sigma
    context_mu = tf.stack(patch_mu, axis=1)
    context_sigma = tf.stack(patch_sigma, axis=1)

    # Caching is not yet implemented for Keras.
    decode_caches = [None for _ in range(self.x)]

    normed_inputs = revin(
        patched_inputs, context_mu, context_sigma, reverse=False
    )
    normed_inputs = tf.where(patched_masks, 0.0, normed_inputs)
    (_, _, normed_outputs, normed_quantile_spread), decode_caches = self(
        normed_inputs, patched_masks, decode_caches
    )
    renormed_outputs = tf.reshape(
        revin(normed_outputs, context_mu, context_sigma, reverse=True),
        (batch_size, -1, self.o, self.q),
    )
    renormed_quantile_spread = tf.reshape(
        revin(normed_quantile_spread, context_mu, context_sigma, reverse=True),
        (batch_size, -1, self.os, self.q),
    )[:, -1, ...]

    # Autogressive decode
    ar_outputs = []
    last_renormed_output = renormed_outputs[:, -1, :, self.aridx]

    for _ in range(num_decode_steps):
      new_patched_input = tf.reshape(
          last_renormed_output, (batch_size, self.m, self.p)
      )
      new_mask = tf.zeros_like(new_patched_input, dtype=tf.bool)

      n, mu, sigma = last_n, last_mu, last_sigma
      new_mus, new_sigmas = [], []
      for i in range(self.m):
        (n, mu, sigma), _ = util.update_running_stats(
            n, mu, sigma, new_patched_input[:, i], new_mask[:, i]
        )
        new_mus.append(mu)
        new_sigmas.append(sigma)
      last_n, last_mu, last_sigma = n, mu, sigma
      new_mu = tf.stack(new_mus, axis=1)
      new_sigma = tf.stack(new_sigmas, axis=1)

      new_normed_input = revin(
          new_patched_input, new_mu, new_sigma, reverse=False
      )
      (_, _, new_normed_output, _), decode_caches = self(
          new_normed_input, new_mask, decode_caches
      )

      new_renormed_output = tf.reshape(
          revin(new_normed_output, new_mu, new_sigma, reverse=True),
          (batch_size, self.m, self.o, self.q),
      )
      ar_outputs.append(new_renormed_output[:, -1, ...])
      last_renormed_output = new_renormed_output[:, -1, :, self.aridx]

    if num_decode_steps > 0:
      ar_renormed_outputs = tf.stack(ar_outputs, axis=1)
    else:
      ar_renormed_outputs = None

    return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs


class TimesFM_2p5_200M_keras(timesfm_2p5_base.TimesFM_2p5):
  """Keras implementation of TimesFM 2.5 with 200M parameters."""

  model: tf.keras.Model = TimesFM_2p5_200M_keras_module()

  def load_checkpoint(self, path: str):
    # Keras model loading is typically handled by `model.load_weights(path)`.
    # The safetensors format from the PyTorch version would require a conversion
    # script. For now, this is a placeholder.
    print(f"Loading checkpoint from: {path}")
    # self.model.load_weights(path) # This would be the typical Keras way.

  def compile(self, forecast_config: configs.ForecastConfig, **kwargs):
    self.global_batch_size = forecast_config.per_core_batch_size  # No multi-device support assumed

    fc = forecast_config

    if fc.max_context % self.model.p != 0:
      new_context = math.ceil(fc.max_context / self.model.p) * self.model.p
      fc.max_context = new_context
    if fc.max_horizon % self.model.o != 0:
      new_horizon = math.ceil(fc.max_horizon / self.model.o) * self.model.o
      fc.max_horizon = new_horizon
    if fc.max_context + fc.max_horizon > self.model.config.context_limit:
      raise ValueError(
          "Context + horizon must be less than the context limit."
      )
    if fc.use_continuous_quantile_head and (fc.max_horizon > self.model.os):
      raise ValueError(
          "Continuous quantile head is not supported for horizons >"
          f" {self.model.os}."
      )
    self.forecast_config = fc

    @tf.function
    def _compiled_decode(horizon, inputs, masks):
      if horizon > fc.max_horizon:
        # tf.function doesn't support raising exceptions with dynamic values.
        # This check is more for the Python wrapper.
        pass

      inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
      masks = tf.convert_to_tensor(masks, dtype=tf.bool)
      batch_size = tf.shape(inputs)[0]

      if fc.infer_is_positive:
        is_positive = tf.reduce_all(inputs >= 0, axis=-1, keepdims=True)
      else:
        is_positive = None

      if fc.normalize_inputs:
        mu = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        sigma = tf.math.reduce_std(inputs, axis=-1, keepdims=True)
        inputs = revin(inputs, mu, sigma, reverse=False)
      else:
        mu, sigma = None, None

      pf_outputs, quantile_spreads, ar_outputs = self.model.decode(
          forecast_config.max_horizon, inputs, masks
      )
      
      if ar_outputs is not None:
        full_forecast = tf.concat(
            [
                pf_outputs[:, -1, ...],
                tf.reshape(ar_outputs, (batch_size, -1, self.model.q)),
            ],
            axis=1,
        )
      else:
        full_forecast = pf_outputs[:, -1, ...]


      if fc.use_continuous_quantile_head:
        for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
          # This is a bit tricky in TF graph mode. A loop is better.
          pass

      full_forecast = full_forecast[:, :horizon, :]

      if fc.return_backcast:
        full_backcast = tf.reshape(
            pf_outputs[:, :-1, : self.model.p, :],
            (batch_size, -1, self.model.q),
        )
        full_forecast = tf.concat([full_backcast, full_forecast], axis=1)

      if fc.fix_quantile_crossing:
        # This is also tricky in TF graph mode.
        pass

      if fc.normalize_inputs:
        full_forecast = revin(full_forecast, mu, sigma, reverse=True)

      if is_positive is not None:
        full_forecast = tf.where(
            is_positive[..., tf.newaxis],
            tf.maximum(full_forecast, tf.zeros_like(full_forecast)),
            full_forecast,
        )

      return full_forecast[..., 5], full_forecast

    self.compiled_decode = _compiled_decode