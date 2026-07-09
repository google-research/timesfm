# Fine-Tuning TimesFM 2.5 with LoRA

Parameter-efficient fine-tuning of
[TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-transformers)
using **HuggingFace Transformers** and **PEFT (LoRA)**.

This approach is based on the fine-tuning workflow by
[@kashif](https://github.com/kashif) at HuggingFace
([notebook](https://github.com/huggingface/notebooks/blob/main/examples/timesfm2_5.ipynb)).

## How It Works

TimesFM 2.5 is available as a standard
[Transformers](https://github.com/huggingface/transformers) model
(`TimesFm2_5ModelForPrediction`). This means it supports the full Transformers
ecosystem out of the box, including:

- **PEFT adapters** — LoRA, QLoRA, etc. via the
  [`peft`](https://github.com/huggingface/peft) library
- **All attention backends** — eager, SDPA, Flash Attention 2/3, Flex Attention
- **Standard `from_pretrained` / `save_pretrained` workflow**

The model's forward pass natively computes a training loss when `future_values`
are provided, so fine-tuning requires nothing more than a standard PyTorch
training loop.

## Quick Start

### Install

```bash
pip install transformers accelerate peft pandas pyarrow scikit-learn
```

### Train

```bash
# Fine-tune with default settings on the retail sales dataset
python finetune_lora.py

# Custom hyperparameters
python finetune_lora.py \
    --epochs 20 \
    --batch_size 64 \
    --lr 5e-5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --context_len 64 \
    --horizon_len 13 \
    --output_dir my-retail-adapter
```

### Evaluate

```bash
# Evaluate a previously trained adapter (skip training)
python finetune_lora.py --eval_only --output_dir timesfm2_5-retail-lora
```

## Key Concepts

### No External Normalisation

TimesFM 2.5 applies its own internal instance normalisation (RevIN). **Do not**
normalise your data externally — feed raw values and let the model handle it.

### Random Window Sampling

Following [Chronos-2](https://github.com/amazon-science/chronos-forecasting),
each training example is a random `(context, horizon)` window sliced from one of
the input series. This is more data-efficient than always using the same
fixed window per series.

### LoRA Target Modules

Using `target_modules="all-linear"` applies LoRA to every linear layer in the
model. With `r=4` this adds only ~0.6% trainable parameters (~1.4M out of
~232M), which is enough to meaningfully adapt the model to a new domain.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_id` | `google/timesfm-2.5-200m-transformers` | HuggingFace model ID |
| `--context_len` | `64` | Context length for training windows |
| `--horizon_len` | `13` | Forecast horizon in time steps |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--lora_r` | `4` | LoRA rank |
| `--lora_alpha` | `8` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--num_samples` | `5000` | Random training windows to pre-sample |
| `--output_dir` | `timesfm2_5-retail-lora` | Where to save the adapter |
| `--seed` | `42` | Random seed |
| `--eval_only` | — | Skip training; evaluate existing adapter |

## Acknowledgements

The Transformers integration and fine-tuning approach were developed by
[@kashif](https://github.com/kashif) at HuggingFace. See the original notebook:
<https://github.com/huggingface/notebooks/blob/main/examples/timesfm2_5.ipynb>
