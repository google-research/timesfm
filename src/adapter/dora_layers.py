# Copyright 2024 The Google Research Authors.
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

from jax import numpy as jnp
from praxis import base_layer, pytypes
from praxis.layers.attentions import AttentionProjection, CombinedQKVProjectionLayer
from praxis.layers.linears import Linear

WeightInit = base_layer.WeightInit
template_field = base_layer.template_field
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor


class DoraTheta(base_layer.Theta):
    def __init__(self, module):
        self.module = module

    def _dora_initialized(self):
        if (
            self.module.has_variable("params", "lora_a")
            and self.module.has_variable("params", "lora_b")
            and self.module.has_variable("params", "dora_m")
            and "lora_a" in self.module._weight_hparams
            and "lora_b" in self.module._weight_hparams
            and "dora_m" in self.module._weight_hparams
        ):
            return True
        else:
            return False

    def _dorafy_var(self, var):
        lora_a = super().__getattr__("lora_a")
        lora_b = super().__getattr__("lora_b")
        dora_m = super().__getattr__("dora_m")

        new_var = self.module.einsum("...dr,...nr->...dn", lora_a, lora_b)
        new_var = jnp.reshape(new_var, var.shape)

        new_var += var

        column_norm = jnp.linalg.norm(new_var, ord=2, axis=0, keepdims=True)
        norm_adapted = new_var / column_norm
        w = dora_m * norm_adapted
        return w

    def __getattr__(self, k):
        var = super().__getattr__(k)
        if not self._dora_initialized():
            return var

        if k == "w":
            return self._dorafy_var(var)

        return var

    def __getitem__(self, k):
        var = super().__getattr__(k)
        if not self._dora_initialized():
            return var

        if k == "w":
            return self._dorafy_var(var)

        return var


class DoraThetaDescriptor:
    """Dot syntax accession descriptor."""

    def __get__(self, obj, objtype=None):
        return DoraTheta(obj)


class DoraLinear(Linear):
    rank: int = 0
    lora_init: WeightInit | None = None
    theta = DoraThetaDescriptor()

    def setup(self) -> None:
        lora_init = self.lora_init if self.lora_init else self.weight_init

        super().setup()
        self.create_variable(
            "lora_a",
            WeightHParams(
                shape=[self.input_dims, self.rank],
                init=lora_init,
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[None, None],
            ),
        )
        self.create_variable(
            "lora_b",
            WeightHParams(
                shape=[self.output_dims, self.rank],
                init=WeightInit.Constant(scale=0.0),
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[None, None],
            ),
        )
        self.create_variable(
            "dora_m",
            WeightHParams(
                shape=[1, self.output_dims],
                init=lora_init,
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[None, None],
            ),
        )


class DoraAttentionProjection(AttentionProjection):
    rank: int = 0
    lora_init: WeightInit | None = None
    theta = DoraThetaDescriptor()

    def setup(self) -> None:
        super().setup()
        w_weight_params = self._weight_hparams["w"]
        lora_init = self.lora_init if self.lora_init else w_weight_params.init

        self.create_variable(
            "lora_a",
            WeightHParams(
                shape=[self.input_dim, self.rank],
                init=lora_init,
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[
                    None,
                    None,
                ],
            ),
        )
        self.create_variable(
            "lora_b",
            WeightHParams(
                shape=[self.dim_per_head * self.num_heads, self.rank],
                init=WeightInit.Constant(scale=0.0),
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[
                    None,
                    None,
                ],
            ),
        )
        self.create_variable(
            "dora_m",
            WeightHParams(
                shape=[1, self.num_heads, self.dim_per_head],
                init=lora_init,
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[None, None, None],
            ),
        )


class DoraCombinedQKVProjection(CombinedQKVProjectionLayer):
    rank: int = 0
    lora_init: WeightInit | None = None
    theta = DoraThetaDescriptor()

    def setup(self) -> None:
        super().setup()
        w_weight_params = self._weight_hparams["w"]
        lora_init = self.lora_init if self.lora_init else w_weight_params.init

        self.create_variable(
            "lora_a",
            WeightHParams(
                shape=[3, self.input_dim, self.rank],
                init=lora_init,
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[None, None, None],
            ),
        )
        self.create_variable(
            "lora_b",
            WeightHParams(
                shape=[3, self.dim_per_head * self.num_heads, self.rank],
                init=WeightInit.Constant(scale=0.0),
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[None, None, None],
            ),
        )
        self.create_variable(
            "dora_m",
            WeightHParams(
                shape=[3, 1, self.num_heads, self.dim_per_head],
                init=lora_init,
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=[None, None, None, None],
            ),
        )
