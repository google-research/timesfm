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

"""Dense layers for TimesFM."""

from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping

from .. import configs

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
Num = jaxtyping.Num

ResidualBlockConfig = configs.ResidualBlockConfig
RandomFourierFeaturesConfig = configs.RandomFourierFeaturesConfig


class ResidualBlock(nnx.Module):
  """Residual block with two linear layers and a linear residual connection."""

  def __init__(self, config: ResidualBlockConfig, *, rngs=nnx.Rngs(42)):
    self.config = config
    self.hidden_layer = nnx.Linear(
      in_features=config.input_dims,
      out_features=config.hidden_dims,
      use_bias=config.use_bias,
      rngs=rngs,
    )
    self.output_layer = nnx.Linear(
      in_features=config.hidden_dims,
      out_features=config.output_dims,
      use_bias=config.use_bias,
      rngs=rngs,
    )
    self.residual_layer = nnx.Linear(
      in_features=config.input_dims,
      out_features=config.output_dims,
      use_bias=config.use_bias,
      rngs=rngs,
    )
    if config.activation == "relu":
      self.activation = jax.nn.relu
    elif config.activation == "swish":
      self.activation = jax.nn.swish
    elif config.activation == "none":
      self.activation = lambda x: x
    else:
      raise ValueError(f"Activation: {config.activation} not supported.")

  def __call__(self, x: Float[Array, "b ... i"]) -> Float[Array, "b ... o"]:
    return self.output_layer(
      self.activation(self.hidden_layer(x))
    ) + self.residual_layer(x)


class RandomFourierFeatures(nnx.Module):
  """Random Fourier features layer."""

  __data__ = ("phrase_shifts",)

  def __init__(self, config: RandomFourierFeaturesConfig, *, rngs=nnx.Rngs(42)):
    self.config = config

    if config.output_dims % 4 != 0:
      raise ValueError(
        f"Output dims must be a multiple of 4: {config.output_dims} % 4 != 0."
      )
    num_projected_features = config.output_dims // 4

    self.phase_shifts = nnx.Param(jnp.zeros(shape=(2, num_projected_features)))
    self.projection_layer = nnx.Linear(
      in_features=config.input_dims,
      out_features=num_projected_features,
      use_bias=config.use_bias,
      rngs=rngs,
    )
    self.residual_layer = nnx.Linear(
      in_features=config.input_dims,
      out_features=config.output_dims,
      use_bias=config.use_bias,
      rngs=rngs,
    )

  def __call__(self, x: Float[Array, "b ... i"]) -> Float[Array, "b ... o"]:
    projected = self.projection_layer(x)
    cos_features = jnp.cos(projected)
    sin_features = jnp.sin(projected)
    sq_wave_1 = jnp.sign(jnp.sin(projected + self.phase_shifts[0, :]))
    sq_wave_2 = jnp.sign(jnp.sin(projected + self.phase_shifts[1, :]))
    fourier_features = jnp.concatenate(
      [cos_features, sin_features, sq_wave_1, sq_wave_2], axis=-1
    )
    residual = self.residual_layer(x)
    return fourier_features + residual
