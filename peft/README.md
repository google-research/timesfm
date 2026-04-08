# TimesFM 2.5 — PEFT Fine-Tuning Pipeline

Production-grade **LoRA / DoRA** fine-tuning for
[TimesFM 2.5](https://github.com/google-research/timesfm) (200M PyTorch)
with **multi-GPU** support via PyTorch DDP.

## Features

| Strategy | Description |
|---|---|
| **LoRA** | Low-Rank Adaptation — adds trainable A/B matrices to frozen linear layers ([paper](https://arxiv.org/abs/2106.09685)) |
| **DoRA** | Weight-Decomposed LoRA — decomposes adapted weights into magnitude + direction ([paper](https://arxiv.org/abs/2402.09353)) |
| **Linear Probing** | Train only the output heads (`--train_output_head`) with `--lora_rank 0` |

Additional capabilities:

- **Multi-GPU** via `torchrun` (DDP)
- **Mixed precision** — fp16 or bf16
- **Gradient checkpointing** — trade compute for memory on long contexts
- **Cosine-with-warmup** LR schedule
- **Early stopping** on validation loss
- **Adapter-only** checkpoint saving / loading (safetensors)
- **Weight merging** — fold adapters back into base weights for zero-overhead inference
- **Quantile loss** — optional pinball loss on all 9 quantile channels
- **W&B logging** (opt-in)

## Quick Start

### 1. Install

```bash
# From the repo root
pip install -e ".[torch]"
```

### 2. Prepare Data

Your CSV can be in either format:

- **Long format** — columns: `[id, timestamp, value]`
- **Wide format** — each numeric column is an independent series

### 3. Single-GPU Training

```bash
python -m peft.finetune \
  --data_path data.csv \
  --value_col y \
  --context_len 512 \
  --horizon_len 128 \
  --adapter_type lora \
  --lora_rank 8 \
  --num_epochs 10 \
  --batch_size 32
```

### 4. Multi-GPU Training

```bash
torchrun --nproc_per_node=4 -m peft.finetune \
  --data_path data.csv \
  --value_col y \
  --adapter_type dora \
  --lora_rank 16 \
  --mixed_precision bf16 \
  --gradient_checkpointing
```

### 5. Using the Launch Script

```bash
# Edit environment variables to taste
DATA_PATH=data.csv VALUE_COL=y NUM_GPUS=4 bash peft/finetune.sh
```

## Python API

```python
import numpy as np
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.configs import ForecastConfig

from peft import PEFTConfig, PEFTTrainer, TimeSeriesDataset

# 1. Load pretrained model (no torch.compile for training)
wrapper = TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    torch_compile=False,
)

# 2. Configure PEFT
config = PEFTConfig(
    adapter_type="lora",       # or "dora"
    lora_rank=8,
    lora_alpha=16,
    target_modules="all",      # "all" | "attention" | "ffn"
    learning_rate=1e-4,
    num_epochs=10,
    batch_size=32,
    context_len=512,
    horizon_len=128,
    mixed_precision="bf16",    # "no" | "fp16" | "bf16"
)

# 3. Create datasets
train_series = [np.random.randn(2000).astype(np.float32) for _ in range(100)]
val_series   = [np.random.randn(800).astype(np.float32)  for _ in range(100)]

train_ds = TimeSeriesDataset(train_series, context_len=512, horizon_len=128, stride=32)
val_ds   = TimeSeriesDataset(val_series,   context_len=512, horizon_len=128, stride=128)

# 4. Train
trainer = PEFTTrainer(wrapper.model, config)
history = trainer.fit(train_ds, val_ds)

# 5. Save adapter-only checkpoint (~2 MB for rank-8 LoRA)
trainer.save_adapter("./my_adapter/adapter.safetensors")

# 6. Merge adapter into base model for zero-overhead inference
trainer.merge_adapter()
wrapper.compile(ForecastConfig(max_context=512, max_horizon=128))
point, quantiles = wrapper.forecast(horizon=128, inputs=[my_series])
```

## Loading a Saved Adapter

```python
from peft import PEFTConfig, inject_adapters, load_adapter_weights

wrapper = TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch", torch_compile=False
)

# Must inject adapters with the *same* config before loading weights.
config = PEFTConfig(adapter_type="lora", lora_rank=8, target_modules="all")
inject_adapters(wrapper.model, config)
load_adapter_weights(wrapper.model, "./my_adapter/adapter.safetensors")

# Option A: use with adapters active
# Option B: merge for maximum inference throughput
from peft import merge_adapters
merge_adapters(wrapper.model)
```

## Architecture

TimesFM 2.5 (200M) has 20 transformer layers, each containing:

| Linear Layer | Shape | LoRA params (rank 8) |
|---|---|---|
| `attn.qkv_proj` (fused Q/K/V) | 1280 → 3840 | 40,960 |
| `attn.out` | 1280 → 1280 | 20,480 |
| `ff0` | 1280 → 1280 | 20,480 |
| `ff1` | 1280 → 1280 | 20,480 |

With `target_modules="all"` and `lora_rank=8`:

- **2,048,000** trainable adapter parameters (~1% of the 200M total)
- DoRA adds ~102,400 magnitude parameters (negligible overhead)

## CLI Options

```
python -m peft.finetune --help
```

| Flag | Default | Description |
|---|---|---|
| `--model_id` | `google/timesfm-2.5-200m-pytorch` | HF repo or local path |
| `--data_path` | *(required)* | Path to CSV |
| `--id_col` | `None` | Series identifier column (long format) |
| `--value_col` | `None` | Value column (long format) |
| `--context_len` | 512 | Context window (rounded to multiple of 32) |
| `--horizon_len` | 128 | Prediction horizon (≤ 128) |
| `--adapter_type` | `lora` | `lora` or `dora` |
| `--lora_rank` | 8 | Low-rank dimension |
| `--lora_alpha` | 16 | Scaling factor |
| `--target_modules` | `all` | `all`, `attention`, or `ffn` |
| `--train_output_head` | off | Also train output projections |
| `--num_epochs` | 10 | Training epochs |
| `--batch_size` | 32 | Per-GPU batch size |
| `--learning_rate` | 1e-4 | Peak learning rate |
| `--mixed_precision` | `no` | `no`, `fp16`, or `bf16` |
| `--gradient_checkpointing` | off | Activation checkpointing |
| `--use_quantile_loss` | off | Add pinball loss |
| `--use_wandb` | off | W&B logging |
| `--early_stopping_patience` | 5 | Patience epochs |

## File Layout

```
peft/
├── __init__.py          # Public API
├── adapters.py          # LoRA / DoRA layers + inject / merge / save / load
├── config.py            # PEFTConfig dataclass
├── data.py              # TimeSeriesDataset
├── trainer.py           # PEFTTrainer (DDP, AMP, checkpointing)
├── finetune.py          # CLI entry-point
├── finetune.sh          # Example launch script
└── README.md            # This file
```
