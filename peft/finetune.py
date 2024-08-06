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
Finetune pipeline.
"""
import gc
import logging
import warnings
from datetime import datetime
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import typer
import wandb
from jax import numpy as jnp
from paxml import checkpoint_types, checkpoints, learners, tasks_lib, trainer_lib
from praxis import optimizers, pax_fiddle, py_utils, schedules
from rich import print
from tqdm import tqdm
from typing_extensions import Annotated

from adapter.utils import get_adapter_params, load_adapter_layer
from timesfm import TimesFm, data_loader, patched_decoder

NestedMap = py_utils.NestedMap


warnings.filterwarnings("ignore")
cmdstanpy_logger = logging.getLogger("cmdstanpy")
absl_logger = logging.getLogger("absl")
cmdstanpy_logger.disabled = True
absl_logger.disabled = True

"""
TimesFM model config. These are fixed since pre-training was done 
with this configuration.
"""
INPUT_PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
NUM_LAYERS = 20
MODEL_DIMS = 1280

QUANTILES = list(np.arange(1, 10) / 10.0)
EPS = 1e-7
RANDOM_SEED = 1234


def finetune(
    *,
    model_name: Annotated[
        str, typer.Option(help="Specify the name of the huggingface model.")
    ] = "google/timesfm-1.0-200m",
    checkpoint_path: Annotated[
        str, typer.Option(help="The path to the local model checkpoint.")
    ] = None,
    datetime_col: Annotated[str, typer.Option(help="Column having datetime.")] = "ds",
    ts_cols: Annotated[
        list[str], typer.Option(help="Columns of time-series features.")
    ] = [],
    normalize: Annotated[
        bool, typer.Option(help="Normalize data for eval or not")
    ] = True,
    context_len: Annotated[int, typer.Option(help="Length of the context window")],
    horizon_len: Annotated[int, typer.Option(help="Prediction length.")],
    freq: Annotated[
        str,
        typer.Option(
            ...,
            help="Frequency Map Str",
        ),
    ],
    data_path: Annotated[str, typer.Option(help="Path to dataset csv")],
    boundaries: Annotated[
        Tuple[int, int, int],
        typer.Option(
            help="boundaries of dataset to train, val, test",
        ),
    ] = (0, 0, 0),
    backend: Annotated[str, typer.Option(help="Backend device: cpu, gpu, tpu")],
    batch_size: Annotated[
        int, typer.Option(help="Batch size for the randomly sampled batch")
    ] = 16,
    num_epochs: Annotated[int, typer.Option(help="Number of epochs")],
    learning_rate: Annotated[float, typer.Option(help="adam optimizer learning rate")],
    adam_epsilon: Annotated[float, typer.Option(help="adam optimizer epsilon")],
    adam_clip_threshold: Annotated[
        float, typer.Option(help="adam optimizer clip threshold")
    ],
    cos_initial_decay_value: Annotated[
        float, typer.Option(help="cosine initial decay value")
    ],
    cos_final_decay_value: Annotated[
        float, typer.Option(help="cosine final decay value")
    ],
    cos_decay_steps: Annotated[int, typer.Option(help="Number of cosine decay steps")],
    ema_decay: Annotated[float, typer.Option(help="Exponential moving average decay")],
    early_stop_patience: Annotated[
        int, typer.Option(..., help="Early stopping patience")
    ] = 5,
    use_lora: Annotated[
        bool,
        typer.Option(
            help="Train low rank adapters for stacked transformer block",
        ),
    ] = False,
    lora_rank: Annotated[
        int,
        typer.Option(
            help="LoRA Rank",
        ),
    ] = 8,
    lora_target_modules: Annotated[
        str,
        typer.Option(
            help="LoRA target modules of the transformer block. Allowed values: [all, attention, mlp]"
        ),
    ] = "all",
    use_dora: Annotated[
        bool,
        typer.Option(
            help="Apply DoRA strategy along with LoRA.",
        ),
    ] = False,
    use_linear_probing: Annotated[
        bool,
        typer.Option(
            help="Linear Probing. Train only input/output and embedding params. Freeze params in stack transformer block.",
        ),
    ] = False,
    checkpoint_dir: Annotated[
        str, typer.Option(help="Checkpoint directory")
    ] = "./checkpoints",
    wandb_project: Annotated[
        str, typer.Option(help="Weights & Biases project name")
    ] = "google_timesfm_finetune",
) -> None:
    key = jax.random.PRNGKey(seed=RANDOM_SEED)
    wandb.init(project=wandb_project, config=locals())

    data_df = pd.read_csv(open(data_path, "r"))

    if boundaries == (0, 0, 0):
        # Default boundaries: train 60%, val 20%, test 20%
        boundaries = [
            int(len(data_df) * 0.6),
            int(len(data_df) * 0.8),
            len(data_df) - 1,
        ]

    ts_cols = [col for col in data_df.columns if col != datetime_col]

    dtl = data_loader.TimeSeriesdata(
        data_path=data_path,
        datetime_col=datetime_col,
        num_cov_cols=None,
        cat_cov_cols=None,
        ts_cols=np.array(ts_cols),
        train_range=[0, boundaries[0]],
        val_range=[boundaries[0], boundaries[1]],
        test_range=[boundaries[1], boundaries[2]],
        hist_len=context_len,
        pred_len=horizon_len,
        batch_size=batch_size,
        freq=freq,
        normalize=normalize,
        epoch_len=None,
        holiday=False,
        permute=False,
    )

    train_batches = dtl.tf_dataset(mode="train", shift=1).batch(batch_size)
    val_batches = dtl.tf_dataset(mode="val", shift=horizon_len)

    for tbatch in tqdm(train_batches.as_numpy_iterator()):
        pass

    tfm = TimesFm(
        context_len=context_len,
        horizon_len=horizon_len,
        input_patch_len=INPUT_PATCH_LEN,
        output_patch_len=OUTPUT_PATCH_LEN,
        num_layers=NUM_LAYERS,
        model_dims=MODEL_DIMS,
        backend=backend,
        per_core_batch_size=batch_size,
        quantiles=QUANTILES,
    )

    if checkpoint_path:
        tfm.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_type=checkpoints.CheckpointType.FLAX,
        )
    else:
        tfm.load_from_checkpoint(
            repo_id=model_name,
            checkpoint_type=checkpoints.CheckpointType.FLAX,
        )

    model = pax_fiddle.Config(
        patched_decoder.PatchedDecoderFinetuneModel,
        name="patched_decoder_finetune",
        core_layer_tpl=tfm.model_p,
    )

    if use_lora:
        load_adapter_layer(
            mdl_vars=tfm._train_state.mdl_vars,
            model=model.core_layer_tpl,
            lora_rank=lora_rank,
            lora_target_modules=lora_target_modules,
            use_dora=use_dora,
        )

    @pax_fiddle.auto_config
    def build_learner() -> learners.Learner:
        bprop_variable_inclusion = []
        bprop_variable_exclusion = []
        if use_lora:
            bprop_variable_inclusion.append(r"^.*lora.*$")
            if use_dora:
                bprop_variable_inclusion.append(r"^.*dora.*$")
        elif use_linear_probing:
            bprop_variable_exclusion = [".*/stacked_transformer_layer/.*"]

        return pax_fiddle.Config(
            learners.Learner,
            name="learner",
            loss_name="avg_qloss",
            optimizer=optimizers.Adam(
                epsilon=adam_epsilon,
                clip_threshold=adam_clip_threshold,
                learning_rate=learning_rate,
                lr_schedule=pax_fiddle.Config(
                    schedules.Cosine,
                    initial_value=cos_initial_decay_value,
                    final_value=cos_final_decay_value,
                    total_steps=cos_decay_steps,
                ),
                ema_decay=ema_decay,
            ),
            bprop_variable_exclusion=bprop_variable_exclusion,
            bprop_variable_inclusion=bprop_variable_inclusion,
        )

    task_p = tasks_lib.SingleTask(
        name="ts-learn",
        model=model,
        train=tasks_lib.SingleTask.Train(
            learner=build_learner(),
        ),
    )

    task_p.model.ici_mesh_shape = [1, 1, 1]
    task_p.model.mesh_axis_names = ["replica", "data", "mdl"]

    DEVICES = np.array(jax.devices()).reshape([1, 1, 1])
    jax.sharding.Mesh(DEVICES, ["replica", "data", "mdl"])

    num_devices = jax.local_device_count()
    print(f"num_devices: {num_devices}")
    print(f"device kind: {jax.local_devices()[0].device_kind}")

    jax_task = task_p
    key, init_key = jax.random.split(key)

    def process_train_batch(batch):
        past_ts = batch[0].reshape(batch_size * len(ts_cols), -1)
        actual_ts = batch[3].reshape(batch_size * len(ts_cols), -1)
        return NestedMap(input_ts=past_ts, actual_ts=actual_ts)

    def process_eval_batch(batch):
        past_ts = batch[0]
        actual_ts = batch[3]
        return NestedMap(input_ts=past_ts, actual_ts=actual_ts)

    jax_model_states, _ = trainer_lib.initialize_model_state(
        jax_task,
        init_key,
        process_train_batch(tbatch),
        checkpoint_type=checkpoint_types.CheckpointType.GDA,
    )
    jax_model_states.mdl_vars["params"]["core_layer"] = tfm._train_state.mdl_vars[
        "params"
    ]
    gc.collect()

    jax_task = task_p

    def train_step(states, prng_key, inputs):
        return trainer_lib.train_step_single_learner(jax_task, states, prng_key, inputs)

    def eval_step(states, prng_key, inputs):
        states = states.to_eval_state()
        return trainer_lib.eval_step_single_learner(jax_task, states, prng_key, inputs)

    key, train_key, eval_key = jax.random.split(key, 3)
    train_prng_seed = jax.random.split(train_key, num=jax.local_device_count())
    eval_prng_seed = jax.random.split(eval_key, num=jax.local_device_count())

    p_train_step = jax.pmap(train_step, axis_name="batch")
    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    replicated_jax_states = trainer_lib.replicate_model_state(jax_model_states)

    def reshape_batch_for_pmap(batch, num_devices):
        def _reshape(input_tensor):
            bsize = input_tensor.shape[0]
            residual_shape = list(input_tensor.shape[1:])
            nbsize = bsize // num_devices
            return jnp.reshape(input_tensor, [num_devices, nbsize] + residual_shape)

        return jax.tree.map(_reshape, batch)

    patience = 0
    best_eval_loss = 1e7
    checkpoint_dir = f"{checkpoint_dir}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}"
    for epoch in range(num_epochs):
        if patience >= early_stop_patience:
            print("Early stopping.")
            break
        print(f"Epoch: {epoch + 1}")
        train_its = train_batches.as_numpy_iterator()
        train_losses = []
        for batch in tqdm(train_its):
            tbatch = process_train_batch(batch)
            tbatch = reshape_batch_for_pmap(tbatch, num_devices)
            replicated_jax_states, step_fun_out = p_train_step(
                replicated_jax_states, train_prng_seed, tbatch
            )
            train_losses.append(step_fun_out.loss[0])
            wandb.log({"train_step_loss": step_fun_out.loss[0]})

        avg_train_loss = np.mean(train_losses)

        print("Starting eval.")
        val_its = val_batches.as_numpy_iterator()
        eval_losses = []
        for ev_batch in tqdm(val_its):
            ebatch = process_eval_batch(ev_batch)
            ebatch = reshape_batch_for_pmap(ebatch, num_devices)
            _, step_fun_out = p_eval_step(replicated_jax_states, eval_prng_seed, ebatch)
            eval_losses.append(step_fun_out.loss[0])
            wandb.log({"eval_step_loss": step_fun_out.loss[0]})

        avg_eval_loss = np.mean(eval_losses)

        print(f"Train Loss: {avg_train_loss}, Val Loss: {avg_eval_loss}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_eval_loss,
            }
        )

        if avg_eval_loss < best_eval_loss or np.isnan(avg_eval_loss):
            best_eval_loss = avg_eval_loss
            print("Saving checkpoint.")
            jax_state_for_saving = py_utils.maybe_unreplicate_for_fully_replicated(
                replicated_jax_states
            )
            if use_lora:
                adapter_params = get_adapter_params(
                    params=jax_state_for_saving.mdl_vars,
                    lora_target_modules=lora_target_modules,
                    num_layers=NUM_LAYERS,
                    use_dora=use_dora,
                )
                jax_state_for_saving.mdl_vars["params"] = adapter_params

            checkpoints.save_checkpoint(
                jax_state_for_saving, checkpoint_dir, overwrite=True
            )

            patience = 0
            del jax_state_for_saving
            gc.collect()
        else:
            patience += 1
            print(f"patience: {patience}")
    print("Fine-tuning completed.")


if __name__ == "__main__":
    typer.run(finetune)
