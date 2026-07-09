"""
Example usage of the TimesFM Finetuning Framework.

For single GPU:
python script.py --training_mode=single

For multiple GPUs:
python script.py --training_mode=multi --gpu_ids=0,1,2
"""

import os
from dataclasses import asdict
from os import path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yfinance as yf
from absl import app, flags
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch.utils.data import Dataset

from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import (PatchedTimeSeriesDecoder,
                                             TimesFMConfig)

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "training_mode",
    "single",
    ["single", "multi"],
    'Training mode: "single" for single-GPU or "multi" for multi-GPU training.',
)

flags.DEFINE_list(
    "gpu_ids", ["0"],
    "Comma-separated list of GPU IDs to use for multi-GPU training. Example: 0,1,2"
)

flags.DEFINE_string(
    "local_model_path",
    None,
    "Path to a local .safetensors model file. If provided, overrides Hugging Face download."
)

class TimeSeriesDataset(Dataset):
  """Dataset for time series data compatible with TimesFM."""

  def __init__(self,
               series: np.ndarray,
               context_length: int,
               horizon_length: int,
               freq_type: int = 0):
    """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
    if freq_type not in [0, 1, 2]:
      raise ValueError("freq_type must be 0, 1, or 2")

    self.series = series
    self.context_length = context_length
    self.horizon_length = horizon_length
    self.freq_type = freq_type
    self._prepare_samples()

  def _prepare_samples(self) -> None:
    """Prepare sliding window samples from the time series."""
    self.samples = []
    total_length = self.context_length + self.horizon_length

    for start_idx in range(0, len(self.series) - total_length + 1):
      end_idx = start_idx + self.context_length
      x_context = self.series[start_idx:end_idx]
      x_future = self.series[end_idx:end_idx + self.horizon_length]
      self.samples.append((x_context, x_future))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(
      self, index: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_context, x_future = self.samples[index]

    x_context = torch.tensor(x_context, dtype=torch.float32)
    x_future = torch.tensor(x_future, dtype=torch.float32)

    input_padding = torch.zeros_like(x_context)
    freq = torch.tensor([self.freq_type], dtype=torch.long)

    return x_context, input_padding, freq, x_future


def prepare_datasets(series: np.ndarray,
                     context_length: int,
                     horizon_length: int,
                     freq_type: int = 0,
                     train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
  """
    Prepare training and validation datasets from time series data.

    Args:
        series: Input time series data
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
  train_size = int(len(series) * train_split)
  train_data = series[:train_size]
  val_data = series[train_size:]

  # Create datasets with specified frequency type
  train_dataset = TimeSeriesDataset(train_data,
                                    context_length=context_length,
                                    horizon_length=horizon_length,
                                    freq_type=freq_type)

  val_dataset = TimeSeriesDataset(val_data,
                                  context_length=context_length,
                                  horizon_length=horizon_length,
                                  freq_type=freq_type)

  return train_dataset, val_dataset


def get_model(load_weights: bool = False):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  hparams = TimesFmHparams(
      backend=device,
      per_core_batch_size=32,
      horizon_len=128,
      num_layers=50,
      use_positional_embedding=False,
      context_len=192,
  )
  
  if load_weights:
    if FLAGS.local_model_path:
      tfm_config = TimesFMConfig()
      model = PatchedTimeSeriesDecoder(tfm_config)
      loaded_checkpoint = load_file(FLAGS.local_model_path)
    else:
      repo_id = "google/timesfm-2.0-500m-pytorch"
      tfm = TimesFm(hparams=hparams,
              checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

      tfm_config = tfm._model_config
      model = PatchedTimeSeriesDecoder(tfm_config)
      checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
      loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)

    model.load_state_dict(loaded_checkpoint)
  return model, hparams, tfm_config


def plot_predictions(
    model: TimesFm,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
  """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
      model: Trained TimesFM model
      val_dataset: Validation dataset
      save_path: Path to save the plot
    """
  import matplotlib.pyplot as plt

  model.eval()

  x_context, x_padding, freq, x_future = val_dataset[0]
  x_context = x_context.unsqueeze(0)  # Add batch dimension
  x_padding = x_padding.unsqueeze(0)
  freq = freq.unsqueeze(0)
  x_future = x_future.unsqueeze(0)

  device = next(model.parameters()).device
  x_context = x_context.to(device)
  x_padding = x_padding.to(device)
  freq = freq.to(device)
  x_future = x_future.to(device)

  with torch.no_grad():
    predictions = model(x_context, x_padding.float(), freq)
    predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
    last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

  context_vals = x_context[0].cpu().numpy()
  future_vals = x_future[0].cpu().numpy()
  pred_vals = last_patch_pred[0].cpu().numpy()

  context_len = len(context_vals)
  horizon_len = len(future_vals)

  plt.figure(figsize=(12, 6))

  plt.plot(range(context_len),
           context_vals,
           label="Historical Data",
           color="blue",
           linewidth=2)

  plt.plot(
      range(context_len, context_len + horizon_len),
      future_vals,
      label="Ground Truth",
      color="green",
      linestyle="--",
      linewidth=2,
  )

  plt.plot(range(context_len, context_len + horizon_len),
           pred_vals,
           label="Prediction",
           color="red",
           linewidth=2)

  plt.xlabel("Time Step")
  plt.ylabel("Value")
  plt.title("TimesFM Predictions vs Ground Truth")
  plt.legend()
  plt.grid(True)

  if save_path:
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

  plt.close()


def get_data(context_len: int,
             horizon_len: int,
             freq_type: int = 0) -> Tuple[Dataset, Dataset]:
  df = yf.download("AAPL", start="2010-01-01", end="2019-01-01")
  time_series = df["Close"].values

  train_dataset, val_dataset = prepare_datasets(
      series=time_series,
      context_length=context_len,
      horizon_length=horizon_len,
      freq_type=freq_type,
      train_split=0.8,
  )

  print(f"Created datasets:")
  print(f"- Training samples: {len(train_dataset)}")
  print(f"- Validation samples: {len(val_dataset)}")
  print(f"- Using frequency type: {freq_type}")
  return train_dataset, val_dataset


def single_gpu_example():
  """Basic example of finetuning TimesFM on stock data."""
  model, hparams, tfm_config = get_model(load_weights=True)
  config = FinetuningConfig(batch_size=256,
                            num_epochs=5,
                            learning_rate=1e-4,
                            use_wandb=True,
                            freq_type=1,
                            log_every_n_steps=10,
                            val_check_interval=0.5,
                            use_quantile_loss=True)

  train_dataset, val_dataset = get_data(128,
                                        tfm_config.horizon_len,
                                        freq_type=config.freq_type)
  finetuner = TimesFMFinetuner(model, config)

  print("\nStarting finetuning...")
  results = finetuner.finetune(train_dataset=train_dataset,
                               val_dataset=val_dataset)

  print("\nFinetuning completed!")
  print(f"Training history: {len(results['history']['train_loss'])} epochs")

  plot_predictions(
      model=model,
      val_dataset=val_dataset,
      save_path="timesfm_predictions.png",
  )


def setup_process(rank, world_size, model, config, train_dataset, val_dataset,
                  return_dict):
  """Setup process function with optimized CUDA handling."""
  try:
    if torch.cuda.is_available():
      torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    if not torch.distributed.is_initialized():
      torch.distributed.init_process_group(backend="nccl",
                                           world_size=world_size,
                                           rank=rank)

    finetuner = TimesFMFinetuner(model, config, rank=rank)

    results = finetuner.finetune(train_dataset=train_dataset,
                                 val_dataset=val_dataset)

    if rank == 0:
      return_dict["results"] = results
      plot_predictions(
          model=model,
          val_dataset=val_dataset,
          save_path="timesfm_predictions.png",
      )

  except Exception as e:
    print(f"Error in process {rank}: {str(e)}")
    raise e
  finally:
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()


def multi_gpu_example():
  """Example of finetuning TimesFM using multiple GPUs with optimized spawn."""
  mp.set_start_method("spawn", force=True)

  gpu_ids = [0, 1]
  world_size = len(gpu_ids)

  model, hparams, tfm_config = get_model(load_weights=True)

  # Create config
  config = FinetuningConfig(
      batch_size=256,
      num_epochs=5,
      learning_rate=3e-5,
      use_wandb=True,
      distributed=True,
      gpu_ids=gpu_ids,
      log_every_n_steps=50,
      val_check_interval=0.5,
  )
  train_dataset, val_dataset = get_data(128, tfm_config.horizon_len)
  manager = mp.Manager()
  return_dict = manager.dict()

  # Launch processes
  mp.spawn(
      setup_process,
      args=(world_size, model, config, train_dataset, val_dataset, return_dict),
      nprocs=world_size,
      join=True,
  )

  results = return_dict.get("results", None)
  print("\nFinetuning completed!")
  return results


def main(argv):
  """Main function that selects and runs the appropriate training mode."""

  try:
    if FLAGS.training_mode == "single":
      print("\nStarting single-GPU training...")
      single_gpu_example()
    else:
      gpu_ids = [int(id) for id in FLAGS.gpu_ids]
      print(f"\nStarting multi-GPU training using GPUs: {gpu_ids}...")

      config = FinetuningConfig(
          batch_size=256,
          num_epochs=5,
          learning_rate=3e-5,
          use_wandb=True,
          distributed=True,
          gpu_ids=gpu_ids,
      )

      results = multi_gpu_example(config)
      print("\nMulti-GPU training completed!")

  except Exception as e:
    print(f"Training failed: {str(e)}")
  finally:
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()


if __name__ == "__main__":
  app.run(main)
