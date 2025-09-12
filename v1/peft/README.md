# Fine-Tuning Pipeline

This folder contains a generic fine-tuning pipeline designed to support multiple PEFT fine-tuning strategies.

## Features

- **Supported Fine-Tuning Strategies**:
  - **Full Fine-Tuning**: Adjusts all parameters of the model during training.
  - **[Linear Probing](https://arxiv.org/abs/2302.11939)**: Fine-tunes only the residual blocks and the embedding layer, leaving other parameters unchanged.
  - **[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)**: A memory-efficient method that fine-tunes a small number of parameters by decomposing the weight matrices into low-rank matrices.
  - **[DoRA (Directional LoRA)](https://arxiv.org/abs/2402.09353v4)**: An extension of LoRA that decomposes pre-trained weights into magnitude and direction components. It uses LoRA for directional adaptation, enhancing learning capacity and stability without additional inference overhead.

## Usage
### Fine-Tuning Script
The provided finetune.py script allows you to fine-tune a model with specific configurations. You can customize various parameters to suit your dataset and desired fine-tuning strategy.

Example Usage:

```zsh
source finetune.sh
```
This script runs the finetune.py file with a predefined set of hyperparameters for the model. You can adjust the parameters in the script as needed.

### Available Options
Run the script with the --help flag to see a full list of available options and their descriptions:
```zsh
python3 finetune.py --help
```
Script Configuration
You can modify the following key parameters directly in the finetune.sh script:
Fine-Tuning Strategy: Toggle between full fine-tuning, LoRA \[`--use-lora`\], DoRA [\[`--use-dora`\]], or Linear Probing \[`--use-linear-probing`\].

### Performance Comparison
The figure below compares the performance of LoRA/DoRA against Linear Probing under the following conditions:

<img width="528" alt="image" src="https://github.com/user-attachments/assets/6c9f820b-5865-4821-8014-c346b9d632a5">

- Training data split: 60% train, 20% validation, 20% test.
- Benchmark: context_len=128, horizon_len=96
- Fine-tuning: context_len=128, horizon_len=128
- Black: Best result.
- Blue: Second best result.
