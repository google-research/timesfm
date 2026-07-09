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

"""Normalization layers for TimesFM."""

from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
Num = jaxtyping.Num


class RMSNorm(nnx.Module):
  """RMS normalization."""

  __data__ = ("scale",)

  def __init__(
    self,
    num_features: int,
    *,
    epsilon: float = 1e-6,
    rngs=nnx.Rngs(42),
  ):
    del rngs
    self.scale = nnx.Param(jnp.zeros(shape=(num_features,)))
    self.num_features = num_features
    self.epsilon = epsilon

  def __call__(self, inputs: Float[Array, "b ... d"]) -> Float[Array, "b ... d"]:
    var = jnp.mean(jnp.square(inputs), axis=-1, keepdims=True)
    normed_inputs = inputs * jax.lax.rsqrt(var + self.epsilon)
    normed_inputs *= self.scale
    return normed_inputs


class LayerNorm(nnx.Module):
  """Layer normalization replica of  LayerNorm."""

  __data__ = ("scale", "bias")

  def __init__(self, num_features: int, *, epsilon: float = 1e-6, rngs=nnx.Rngs(42)):
    del rngs
    self.scale = nnx.Param(jnp.ones(shape=(num_features,)))
    self.bias = nnx.Param(jnp.zeros(shape=(num_features,)))
    self.num_features = num_features
    self.epsilon = epsilon

  def __call__(self, inputs: Float[Array, "b ... d"]) -> Float[Array, "b ... d"]:
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
    normed_inputs = (inputs - mean) * jax.lax.rsqrt(var + self.epsilon)
    normed_inputs *= self.scale
    normed_inputs += self.bias
    return normed_inputs
