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

"""
This file provides functionality for loading and merging adapter weights
in timesfm model, specifically for LoRA and DoRA.
LoRA: https://arxiv.org/abs/2106.09685
DoRA: https://arxiv.org/abs/2402.09353v4 
"""

import time

import jax
import jax.numpy as jnp
from paxml import checkpoints, tasks_lib
from paxml.train_states import TrainState
from praxis import pax_fiddle

from adapter.dora_layers import (
    DoraAttentionProjection,
    DoraCombinedQKVProjection,
    DoraLinear,
)
from adapter.lora_layers import (
    LoraAttentionProjection,
    LoraCombinedQKVProjection,
    LoraLinear,
)
from timesfm import TimesFm


def get_adapter_params(
    params: dict, lora_target_modules: str, num_layers: int, use_dora: bool = False
) -> dict:
    """
    Extracts adapter parameters from the given model parameters for saving the checkpoint.

    Args:
        params (dict): The full model parameters.
        lora_target_modules (str): Target modules for LoRA/DoRA adaptation.
        num_layers (int): Number of transformer layers.
        use_dora (bool, optional): Whether DoRA was used or not. Defaults to False.

    Returns:
        dict: A dictionary containing the extracted adapter parameters.
    """
    adapter_params = {}
    for i in range(num_layers):
        layer_key = f"x_layers_{i}"
        adapter_params[layer_key] = {}

        if lora_target_modules in ["all", "mlp"]:
            for ff_layer_key in ["ffn_layer1", "ffn_layer2"]:
                linear = params["params"]["core_layer"]["stacked_transformer_layer"][
                    layer_key
                ]["ff_layer"][ff_layer_key]["linear"]

                lora_a = linear["lora_a"]
                lora_b = linear["lora_b"]

                adapter_params[layer_key][ff_layer_key] = {
                    "lora_a": lora_a,
                    "lora_b": lora_b,
                }

                if use_dora:
                    adapter_params[layer_key][ff_layer_key]["dora_m"] = linear["dora_m"]

        if lora_target_modules in ["all", "attention"]:
            attention = params["params"]["core_layer"]["stacked_transformer_layer"][
                layer_key
            ]["self_attention"]

            for component in ["key", "query", "value", "post"]:
                lora_a = attention[component]["lora_a"]
                lora_b = attention[component]["lora_b"]

                adapter_params[layer_key][component] = {
                    "lora_a": lora_a,
                    "lora_b": lora_b,
                }

                if use_dora:
                    adapter_params[layer_key][component]["dora_m"] = attention[
                        component
                    ]["dora_m"]
    return adapter_params


def load_adapter_checkpoint(
    model: TimesFm,
    adapter_checkpoint_path: str,
    lora_rank: int,
    lora_target_modules: str,
    use_dora: bool,
) -> None:
    """
    Loads an adapter checkpoint and merges it with the original model weights.

    Args:
        model (TimesFm): The model to update.
        adapter_checkpoint_path (str): Path to the adapter checkpoint.
        lora_rank (int): Rank of the LoRA adaptation.
        lora_target_modules (str): Target modules for adaptation.
        use_dora (bool): Whether DoRA was used or not.

    Returns:
        None
    """

    """
    currently loading and initializing the model with adapter layers first and then merging the
    adapter weights to original weights and replacing the adapter layers back to original layer.
    # NOTE: refactor this. there should be a better way to load the LoRA checkpoint.
    """
    model._logging(f"Restoring adapter checkpoint from {adapter_checkpoint_path}.")
    start_time = time.time()
    original_linear_tpl, original_attn_tpl, original_combined_qkv_tpl = (
        load_adapter_layer(
            mdl_vars=model._train_state.mdl_vars,
            model=model._model,
            lora_rank=lora_rank,
            lora_target_modules=lora_target_modules,
            use_dora=use_dora,
        )
    )

    var_weight_hparams = model._model.abstract_init_with_metadata(
        model._get_sample_inputs(), do_eval=True
    )

    adapter_weight_hparams = _get_adapter_weight_params(
        var_weight_hparams=var_weight_hparams,
        lora_target_modules=lora_target_modules,
        num_layers=model._model.stacked_transformer_params_tpl.num_layers,
        use_dora=use_dora,
    )

    adapter_state_partition_specs = tasks_lib.create_state_partition_specs(
        adapter_weight_hparams,
        mesh_shape=model.mesh_shape,
        mesh_axis_names=model.mesh_name,
        discard_opt_states=True,
        learners=None,
    )
    adapter_state_local_shapes = tasks_lib.create_state_unpadded_shapes(
        adapter_weight_hparams,
        discard_opt_states=True,
        learners=None,
    )
    adapter_train_state = checkpoints.restore_checkpoint(
        state_global_shapes=adapter_state_local_shapes,
        checkpoint_dir=adapter_checkpoint_path,
        checkpoint_type=checkpoints.CheckpointType.FLAX,
        state_specs=adapter_state_partition_specs,
        step=None,
    )

    # add adapter weights to the original weights
    _merge_adapter_weights(
        model=model,
        adapter_train_state=adapter_train_state,
        lora_target_modules=lora_target_modules,
        num_layers=model._model.stacked_transformer_params_tpl.num_layers,
        use_dora=use_dora,
    )

    # replace back with the original model layer
    if lora_target_modules in ["all", "mlp"]:
        model._model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.linear_tpl = (
            original_linear_tpl
        )

    if lora_target_modules in ["all", "attention"]:
        model._model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_atten_tpl.proj_tpl = (
            original_attn_tpl
        )
        model._model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_atten_tpl.combined_qkv_proj_tpl = (
            original_combined_qkv_tpl
        )
    model._logging(
        f"Restored adapter checkpoint in {time.time() - start_time:.2f} seconds."
    )

    # jit compile the model
    model.jit_decode()


def _merge_adapter_weights(
    model: TimesFm,
    adapter_train_state: TrainState,
    lora_target_modules: str,
    num_layers: int,
    use_dora: bool,
) -> None:
    """
    Merges adapter weights with the original model weights.

    Args:
        model (TimesFm): The model to update.
        adapter_train_state (TrainState): The adapter's train state.
        lora_target_modules (str): Target modules for adaptation.
        num_layers (int): Number of transformer layers.
        use_dora (bool): Whether DoRA was used or not.
    """
    for i in range(num_layers):
        layer_key = f"x_layers_{i}"

        if lora_target_modules in ["all", "mlp"]:
            for ff_layer_key in ["ffn_layer1", "ffn_layer2"]:
                linear = model._train_state.mdl_vars["params"][
                    "stacked_transformer_layer"
                ][layer_key]["ff_layer"][ff_layer_key]["linear"]

                params = adapter_train_state.mdl_vars[layer_key][ff_layer_key]
                lora_a = params["lora_a"]
                lora_b = params["lora_b"]

                w = linear["w"]

                lora_delta = jnp.einsum("...dr,...nr->...dn", lora_a, lora_b)
                lora_delta = jnp.reshape(lora_delta, w.shape)
                w_prime = w + lora_delta

                if use_dora:
                    dora_m = params["dora_m"]
                    column_norm = jnp.linalg.norm(w_prime, ord=2, axis=0, keepdims=True)
                    norm_adapted = w_prime / column_norm
                    w_prime = dora_m * norm_adapted
                    linear["w"] = w_prime
                    del linear["dora_m"]

                else:
                    linear["w"] = w_prime

                del linear["lora_a"]
                del linear["lora_b"]

        if lora_target_modules in ["all", "attention"]:
            attention = model._train_state.mdl_vars["params"][
                "stacked_transformer_layer"
            ][layer_key]["self_attention"]

            for component in ["key", "query", "value", "post"]:
                params = adapter_train_state.mdl_vars[layer_key][component]
                lora_a = params["lora_a"]
                lora_b = params["lora_b"]

                w = attention[component]["w"]

                lora_delta = jnp.einsum("...dr,...nr->...dn", lora_a, lora_b)
                lora_delta = jnp.reshape(lora_delta, w.shape)
                w_prime = w + lora_delta

                if use_dora:
                    dora_m = params["dora_m"]
                    column_norm = jnp.linalg.norm(w_prime, ord=2, axis=0, keepdims=True)
                    norm_adapted = w_prime / column_norm
                    w_prime = dora_m * norm_adapted
                    attention[component]["w"] = w_prime
                    del attention[component]["dora_m"]

                else:
                    attention[component]["w"] = w_prime

                del attention[component]["lora_a"]
                del attention[component]["lora_b"]


def _get_adapter_weight_params(
    var_weight_hparams: dict, lora_target_modules: str, num_layers: int, use_dora: bool
) -> dict:
    """
    Extracts adapter weight parameters from the given variable weight hyperparameters.

    Args:
        var_weight_hparams (dict): Variable weight hyperparameters.
        lora_target_modules (str): Target modules for adaptation.
        num_layers (int): Number of transformer layers.
        use_dora (bool): Whether DoRA was used or not.

    Returns:
        dict: A dictionary containing the extracted adapter weight parameters.
    """
    adapter_params = {}
    for i in range(num_layers):
        layer = f"x_layers_{i}"
        adapter_params[layer] = {}

        if lora_target_modules in ["all", "mlp"]:
            for ff_layer_key in ["ffn_layer1", "ffn_layer2"]:
                adapter_weight_params = var_weight_hparams["params"][
                    "stacked_transformer_layer"
                ][layer]["ff_layer"][ff_layer_key]["linear"]
                adapter_params[layer][ff_layer_key] = {
                    "lora_a": adapter_weight_params["lora_a"],
                    "lora_b": adapter_weight_params["lora_b"],
                }

                if use_dora:
                    adapter_params[layer][ff_layer_key]["dora_m"] = (
                        adapter_weight_params["dora_m"]
                    )

        if lora_target_modules in ["all", "attention"]:
            for component in ["key", "value", "query", "post"]:
                adapter_weight_params = var_weight_hparams["params"][
                    "stacked_transformer_layer"
                ][layer]["self_attention"][component]
                adapter_params[layer][component] = {
                    "lora_a": adapter_weight_params["lora_a"],
                    "lora_b": adapter_weight_params["lora_b"],
                }

                if use_dora:
                    adapter_params[layer][component]["dora_m"] = adapter_weight_params[
                        "dora_m"
                    ]

    return adapter_params


def load_adapter_layer(
    mdl_vars: dict,
    model: pax_fiddle.Config,
    lora_rank: int,
    lora_target_modules: str,
    use_dora: bool = False,
) -> tuple[pax_fiddle.Config, pax_fiddle.Config]:
    """
    Updates target modules with adapter layers.

    Args:
        mdl_vars (dict): Model variables.
        model (pax_fiddle.Config): Model configuration.
        lora_rank (int): Rank of the LoRA adaptation.
        lora_target_modules (str): Target modules for adaptation.
        use_dora (bool, optional): Whether DoRA was used or not.

    Returns:
        tuple[pax_fiddle.Config, pax_fiddle.Config]: Updated model configurations.
    """
    original_linear_tpl = original_attn_tpl = original_combined_qkv_tpl = None
    if lora_target_modules in ["all", "mlp"]:
        original_linear_tpl = (
            model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.linear_tpl
        )
        adapter_linear_tpl = (
            pax_fiddle.Config(
                DoraLinear,
                rank=lora_rank,
            )
            if use_dora
            else pax_fiddle.Config(
                LoraLinear,
                rank=lora_rank,
            )
        )
        adapter_linear_tpl.copy_fields_from(original_linear_tpl)
        model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.linear_tpl = (
            adapter_linear_tpl
        )

    if lora_target_modules in ["all", "attention"]:
        original_attn_tpl = (
            model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_atten_tpl.proj_tpl
        )

        adapter_attn_tpl = (
            pax_fiddle.Config(DoraAttentionProjection, rank=lora_rank)
            if use_dora
            else pax_fiddle.Config(LoraAttentionProjection, rank=lora_rank)
        )
        adapter_attn_tpl.copy_fields_from(original_attn_tpl)

        original_combined_qkv_tpl = (
            model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_atten_tpl.combined_qkv_proj_tpl
        )

        adapter_combined_qkv_tpl = (
            pax_fiddle.Config(DoraCombinedQKVProjection, rank=lora_rank)
            if use_dora
            else pax_fiddle.Config(LoraCombinedQKVProjection, rank=lora_rank)
        )
        adapter_combined_qkv_tpl.copy_fields_from(original_combined_qkv_tpl)

        model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_atten_tpl.proj_tpl = (
            adapter_attn_tpl
        )
        model.stacked_transformer_params_tpl.transformer_layer_params_tpl.tr_atten_tpl.combined_qkv_proj_tpl = (
            adapter_combined_qkv_tpl
        )

    # initialize and add adapter weights
    _initialize_adapter_params(
        mdl_vars=mdl_vars,
        num_layers=model.stacked_transformer_params_tpl.num_layers,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        use_dora=use_dora,
    )

    return original_linear_tpl, original_attn_tpl, original_combined_qkv_tpl


def _initialize_adapter_params(
    mdl_vars: dict,
    num_layers,
    lora_rank: int,
    lora_target_modules: str,
    use_dora: bool = False,
    seed: int = 1234,
) -> dict:
    """
    Initializes and adds adapter parameters to target modules.

    Args:
        mdl_vars (dict): Model variables.
        num_layers (int): Number of transformer layers.
        lora_rank (int): Rank of the LoRA adaptation.
        lora_target_modules (str): Target modules for adaptation.
        use_dora (bool, optional): Whether DoRA was used or not.
        seed (int, optional): Random seed for initialization. Defaults to 1234.

    Returns:
        dict: Updated model variables with initialized adapter parameters.
    """
    for i in range(num_layers):
        layer_key = f"x_layers_{i}"
        if lora_target_modules in ["all", "mlp"]:
            for ff_layer_key in ["ffn_layer1", "ffn_layer2"]:
                linear = mdl_vars["params"]["stacked_transformer_layer"][layer_key][
                    "ff_layer"
                ][ff_layer_key]["linear"]
                original_w = linear["w"]
                input_dim, output_dim = original_w.shape
                std_dev = 1 / jnp.sqrt(lora_rank)

                normal_initializer = jax.nn.initializers.normal(std_dev)
                lora_a = normal_initializer(
                    jax.random.key(seed), (input_dim, lora_rank), jnp.float32
                )
                lora_b = jnp.zeros((output_dim, lora_rank))

                linear["lora_a"] = lora_a
                linear["lora_b"] = lora_b

                if use_dora:
                    norm = jnp.linalg.norm(original_w, ord=2, axis=0, keepdims=True)
                    linear["dora_m"] = norm

        if lora_target_modules in ["all", "attention"]:
            attention = mdl_vars["params"]["stacked_transformer_layer"][layer_key][
                "self_attention"
            ]

            for component in ["key", "query", "value", "post"]:
                original_w = attention[component]["w"]
                w_dim = original_w.shape[0]
                std_dev = 1 / jnp.sqrt(lora_rank)

                normal_initializer = jax.nn.initializers.normal(std_dev)
                lora_a = normal_initializer(
                    jax.random.key(seed), (w_dim, lora_rank), jnp.float32
                )
                lora_b = jnp.zeros((w_dim, lora_rank))

                attention[component]["lora_a"] = lora_a
                attention[component]["lora_b"] = lora_b

                if use_dora:
                    norm = jnp.linalg.norm(
                        original_w, ord=2, axis=0, keepdims=True
                    ).astype(jnp.float32)
                    attention[component]["dora_m"] = norm
    return mdl_vars
