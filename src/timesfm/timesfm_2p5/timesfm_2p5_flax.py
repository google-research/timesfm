"""TimesFM model for TMax."""

import functools
import logging
import math
import time
from typing import Any, Callable

import einshape
from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping
import typeguard

from google3.learning.timeseries.scalar.timesfm.tmax.flax import dense
from google3.learning.timeseries.scalar.timesfm.tmax.flax import transformer
from google3.learning.timeseries.scalar.timesfm.tmax.flax import util
from google3.learning.timeseries.scalar.timesfm.tmax import configs
from google3.learning.timeseries.scalar.timesfm.tmax.timesfm_2p5 import timesfm_2p5_base

jax_einshape = einshape.jax_einshape
scan = util.scan_along_axis
revin = util.revin

Float = jaxtyping.Float
Bool = jaxtyping.Bool
Array = jaxtyping.Array


@nnx.vmap(in_axes=(None, 0), out_axes=0)
def _create_stacked_transformers(
    config: configs.StackedTransformersConfig, key: jax.Array
):
  return transformer.Transformer(config.transformer, rngs=nnx.Rngs(key))


@nnx.scan(in_axes=(0, nnx.Carry, None, 0), out_axes=(nnx.Carry, 0))
def _apply_stacked_transformers(
    model: transformer.Transformer,
    x: Float[Array, "b n d"],
    m: Float[Array, "b n"],
    decode_cache: util.DecodeCache | None = None,
) -> Float[Array, "b n d"]:
  return model(x, m, decode_cache=decode_cache)


class TimesFM_2p5_200M_flax_module(nnx.Module):  # pylint: disable=invalid-name
  """TimesFM 2.5 with 200M parameters."""

  config = timesfm_2p5_base.TimesFM_2p5_200M_Definition()
  decode_index: int = 5
  compiled_decode: Callable[..., Any] | None = None
  backend: str = "cpu"
  context: int = 0
  horizon: int = 0
  per_core_batch_size: int = 0

  def __init__(self):
    # Shorcuts.
    self.p = self.config.input_patch_len  # 32
    self.o = self.config.output_patch_len  # 128
    self.m = self.o // self.p  # 4
    self.x = self.config.stacked_transformers.num_layers  # 20
    self.h = self.config.stacked_transformers.transformer.num_heads  # 16
    self.md = self.config.stacked_transformers.transformer.model_dims  # 1280
    self.hd = self.md // self.h  # 80
    self.q = len(self.config.quantiles) + 1

    # Layers.
    self.tokenizer = dense.ResidualBlock(self.config.tokenizer)
    self.stacked_xf = _create_stacked_transformers(
        self.config.stacked_transformers,
        jax.random.split(jax.random.key(42), self.x),
    )
    self.output_projection_point = dense.ResidualBlock(
        self.config.output_projection_point
    )
    self.output_projection_quantiles = dense.ResidualBlock(
        self.config.output_projection_quantiles
    )

  def __call__(
      self,
      inputs: Float[Array, "b n p"],
      masks: Bool[Array, "b n p"],
      decode_cache: util.DecodeCache | None = None,
  ):
    tokenizer_inputs = jnp.concatenate(
        [inputs, masks.astype(inputs.dtype)], axis=-1
    )
    input_embeddings = self.tokenizer(tokenizer_inputs)
    if decode_cache is None:
      decode_cache = [None] * self.x
    output_embeddings, decode_cache = _apply_stacked_transformers(
        self.stacked_xf, input_embeddings, masks[..., -1], decode_cache
    )
    output_ts = self.output_projection_point(output_embeddings)
    return (input_embeddings, output_embeddings, output_ts), decode_cache

  @classmethod
  def load_checkpoint(cls, path: str):
    pass

  def compile(
      self,
      context: int,
      horizon: int,
      per_core_batch_size: int = 1,
      backend: str = "cpu",
  ):
    if context % self.p != 0:
      logging.info(
          "When compiling, context needs to be multiple of the patch size %d."
          " Modifying context to %d.",
          self.p,
          context := math.ceil(context / self.p) * self.p,
      )
    if horizon % self.o != 0:
      logging.info(
          "When compiling, horizon needs to be multiple of the output patch"
          " size %d. Modifying horizon to %d.",
          self.o,
          horizon := math.ceil(horizon / self.o) * self.o,
      )

    self.context = context
    self.horizon = horizon
    self.per_core_batch_size = per_core_batch_size
    self.backend = backend

    @nnx.pmap(
        in_axes=(None, None, 0, 0),
        out_axes=(0, 0),
        devices=jax.devices(self.backend),
        axis_size=len(jax.devices(self.backend)),
        static_broadcasted_argnums=(1,),
        axis_name="global_batch",
    )
    def _compiled_decode(module, horizon, inputs, masks):
      return module.decode(horizon, inputs, masks)

    logging.info("Compiling decode...")
    start_time = time.time()
    self.compiled_decode = functools.partial(_compiled_decode, self)
    _ = self.compiled_decode(
        self.horizon,
        jnp.zeros(
            shape=(
                len(jax.devices(self.backend)),
                self.per_core_batch_size,
                self.context,
            )
        ),
        jnp.zeros(
            shape=(
                len(jax.devices(self.backend)),
                self.per_core_batch_size,
                self.context,
            ),
            dtype=jnp.bool,
        ),
    )
    logging.info(
        "Done compiling decode in %.4f seconds.", time.time() - start_time
    )

  @nnx.jit(static_argnames=("horizon",))
  def decode(self, horizon: int, inputs, masks):
    batch_size, context = inputs.shape[0], inputs.shape[1]
    num_decode_steps = (horizon - 1) // self.o
    num_input_patches = context // self.p
    decode_cache_size = num_input_patches + num_decode_steps * self.m

    # Prefill
    patched_inputs = jax_einshape("b(np)->bnp", inputs, b=batch_size, p=self.p)
    patched_masks = jax_einshape("b(np)->bnp", masks, b=batch_size, p=self.p)
    (last_n, last_mu, last_sigma), (_, context_mu, context_sigma) = scan(
        lambda carry, xs: util.update_running_stats(*carry, *xs),
        init=(zero := jnp.zeros(shape=(batch_size)), zero, zero),
        xs=(patched_inputs, patched_masks),
        axis=1,
    )
    decode_cache = util.DecodeCache(
        next_index=jnp.zeros(shape=(self.x, batch_size), dtype=jnp.int32),
        num_masked=jnp.zeros(shape=(self.x, batch_size), dtype=jnp.int32),
        key=jnp.zeros(
            shape=(self.x, batch_size, decode_cache_size, self.h, self.hd)
        ),
        value=jnp.zeros(
            shape=(self.x, batch_size, decode_cache_size, self.h, self.hd)
        ),
    )
    normed_inputs = revin(
        patched_inputs, context_mu, context_sigma, reverse=False
    )
    normed_inputs = jnp.where(patched_masks, 0.0, normed_inputs)
    (_, _, normed_outputs), decode_cache = self(
        normed_inputs, patched_masks, decode_cache
    )
    renormed_outputs = jax_einshape(
        "bn(oq)->bnoq",
        revin(normed_outputs, context_mu, context_sigma, reverse=True),
        o=self.o,
        q=self.q,
    )

    # Autogressive decode
    @nnx.scan(in_axes=(None, nnx.Carry, 0), out_axes=(nnx.Carry, 1))
    def _ar_decode(module, carry, unused_iter):
      last_renormed_output, (last_n, last_mu, last_sigma), decode_cache = carry
      new_patched_input = jax_einshape(
          "b(mp)->bmp", last_renormed_output, m=module.m, p=module.p
      )
      new_mask = jnp.zeros_like(new_patched_input, dtype=jnp.bool)
      carry_stats, (_, new_mu, new_sigma) = scan(
          lambda carry, xs: util.update_running_stats(*carry, *xs),
          init=(last_n, last_mu, last_sigma),
          xs=(new_patched_input, new_mask),
          axis=1,
      )
      new_normed_input = revin(
          new_patched_input, new_mu, new_sigma, reverse=False
      )
      (_, _, new_normed_output), decode_cache = module(
          new_normed_input, new_mask, decode_cache
      )
      new_renormed_output = jax_einshape(
          "bm(oq)->bmoq",
          revin(new_normed_output, new_mu, new_sigma, reverse=True),
          o=module.o,
          q=module.q,
      )[..., -1, :, :]

      return (
          (
              new_renormed_output[..., module.decode_index],
              carry_stats,
              decode_cache,
          ),
          new_renormed_output,
      )

    _, ar_renormed_outputs = _ar_decode(
        self,
        (
            renormed_outputs[..., -1, :, self.decode_index],
            (last_n, last_mu, last_sigma),
            decode_cache,
        ),
        jnp.arange(num_decode_steps),
    )

    return renormed_outputs, ar_renormed_outputs
