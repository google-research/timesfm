#!/usr/bin/env bash
# ============================================================================
# Example launch script for TimesFM 2.5 PEFT fine-tuning.
#
# Single GPU:
#   bash peft/finetune.sh
#
# Multi-GPU (e.g. 4 GPUs):
#   NUM_GPUS=4 bash peft/finetune.sh
# ============================================================================

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"

# --- Data -------------------------------------------------------------------
DATA_PATH="${DATA_PATH:-data.csv}"       # path to your CSV
ID_COL="${ID_COL:-}"                     # series-id column (long format), leave empty for wide
VALUE_COL="${VALUE_COL:-}"               # value column (long format), leave empty for wide
CONTEXT_LEN="${CONTEXT_LEN:-512}"
HORIZON_LEN="${HORIZON_LEN:-128}"
STRIDE="${STRIDE:-32}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"

# --- Adapter ----------------------------------------------------------------
ADAPTER_TYPE="${ADAPTER_TYPE:-lora}"      # lora | dora
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
TARGET_MODULES="${TARGET_MODULES:-all}"   # all | attention | ffn
NUM_ADAPTER_LAYERS="${NUM_ADAPTER_LAYERS:-4}"  # 0=all 20, advisor recommends 2-4

# --- Training ---------------------------------------------------------------
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"  # no | fp16 | bf16

# --- Logging / checkpoint ---------------------------------------------------
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./peft_checkpoints}"

# ============================================================================

CMD_ARGS=(
  peft/finetune.py
  --data_path "$DATA_PATH"
  --context_len "$CONTEXT_LEN"
  --horizon_len "$HORIZON_LEN"
  --stride "$STRIDE"
  --val_split "$VAL_SPLIT"
  --adapter_type "$ADAPTER_TYPE"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --target_modules "$TARGET_MODULES"
  --num_adapter_layers "$NUM_ADAPTER_LAYERS"
  --train_output_head
  --num_epochs "$NUM_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LR"
  --mixed_precision "$MIXED_PRECISION"
  --checkpoint_dir "$CHECKPOINT_DIR"
)

# Optional columns.
[[ -n "$ID_COL" ]]    && CMD_ARGS+=(--id_col "$ID_COL")
[[ -n "$VALUE_COL" ]] && CMD_ARGS+=(--value_col "$VALUE_COL")

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching multi-GPU training on $NUM_GPUS GPUs …"
  torchrun --nproc_per_node="$NUM_GPUS" "${CMD_ARGS[@]}"
else
  echo "Launching single-GPU training …"
  python "${CMD_ARGS[@]}"
fi
