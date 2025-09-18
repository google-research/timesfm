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
