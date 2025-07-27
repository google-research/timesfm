"""
TimesFM Finetuner: A flexible framework for finetuning TimesFM models on custom datasets.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from tqdm.auto import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from timesfm.pytorch_patched_decoder import create_quantiles

import wandb


# --------------------------- Logging utils --------------------------- #
class MetricsLogger(ABC):
  @abstractmethod
  def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    pass

  @abstractmethod
  def close(self) -> None:
    pass


class WandBLogger(MetricsLogger):
  def __init__(self, project: str, config: Dict[str, Any], rank: int = 0):
    self.rank = rank
    if rank == 0:
      wandb.init(project=project, config=config)

  def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if self.rank == 0:
      wandb.log(metrics, step=step)

  def close(self) -> None:
    if self.rank == 0:
      wandb.finish()


# --------------------------- DDP helper --------------------------- #
class DistributedManager:
  def __init__(
      self,
      world_size: int,
      rank: int,
      master_addr: str = "localhost",
      master_port: str = "12358",
      backend: str = "nccl",
  ):
    self.world_size = world_size
    self.rank = rank
    self.master_addr = master_addr
    self.master_port = master_port
    self.backend = backend

  def setup(self) -> None:
    os.environ["MASTER_ADDR"] = self.master_addr
    os.environ["MASTER_PORT"] = self.master_port
    if not dist.is_initialized():
      dist.init_process_group(backend=self.backend,
                              world_size=self.world_size,
                              rank=self.rank)

  def cleanup(self) -> None:
    if dist.is_initialized():
      dist.destroy_process_group()


# --------------------------- Config --------------------------- #
@dataclass
class FinetuningConfig:
  batch_size: int = 32
  num_epochs: int = 20
  learning_rate: float = 1e-4
  weight_decay: float = 0.01
  freq_type: int = 0
  use_quantile_loss: bool = False
  quantiles: Optional[List[float]] = None

  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  distributed: bool = False
  gpu_ids: List[int] = field(default_factory=lambda: [0])
  master_port: str = "12358"
  master_addr: str = "localhost"

  use_wandb: bool = False
  wandb_project: str = "timesfm-finetuning"

  log_every_n_steps: int = 50
  val_check_interval: float = 0.5

  # ------------- NEW/ADDED ------------- #
  progress_bar: bool = True                     # show tqdm bars
  checkpoint_dir: Optional[str] = "checkpoints" # where to save ckpts
  save_best: bool = True                        # save best val model
  save_every_epoch: bool = False                # additionally save each epoch
  best_ckpt_name: str = "best.ckpt"
  last_ckpt_name: str = "last.ckpt"


# --------------------------- Finetuner --------------------------- #
class TimesFMFinetuner:
  def __init__(
      self,
      model: nn.Module,
      config: FinetuningConfig,
      rank: int = 0,
      loss_fn: Optional[Callable] = None,
      logger: Optional[logging.Logger] = None,
  ):
    self.model = model
    self.config = config
    self.rank = rank
    self.logger = logger or logging.getLogger(__name__)
    self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    self.loss_fn = loss_fn or (lambda x, y: torch.mean((x - y.squeeze(-1))**2))

    if config.use_wandb:
      self.metrics_logger = WandBLogger(config.wandb_project, config.__dict__, rank)

    if config.distributed:
      self.dist_manager = DistributedManager(
          world_size=len(config.gpu_ids),
          rank=rank,
          master_addr=config.master_addr,
          master_port=config.master_port,
      )
      self.dist_manager.setup()
      self.model = self._setup_distributed_model()
    else:
      self.model = self.model.to(self.device)

    # Create checkpoint dir
    if self.rank == 0 and self.config.checkpoint_dir:
      os.makedirs(self.config.checkpoint_dir, exist_ok=True)

  # --------- Public API --------- #
  def finetune(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, Any]:
    """Train the model and return history."""
    train_loader = self._create_dataloader(train_dataset, is_train=True)
    val_loader = self._create_dataloader(val_dataset, is_train=False)

    optimizer = torch.optim.Adam(self.model.parameters(),
                                 lr=self.config.learning_rate,
                                 weight_decay=self.config.weight_decay)

    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    best_val = float("inf")
    best_path = None

    self.logger.info(f"Starting training for {self.config.num_epochs} epochs...")
    self.logger.info(f"Training samples: {len(train_dataset)}")
    self.logger.info(f"Validation samples: {len(val_dataset)}")

    try:
      for epoch in range(self.config.num_epochs):
        train_loss = self._train_epoch(train_loader, optimizer, epoch)
        val_loss = self._validate(val_loader, epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch": epoch + 1,
        }
        if self.config.use_wandb:
          self.metrics_logger.log_metrics(metrics)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        if self.rank == 0:
          self.logger.info(
              f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
          )

        # --- Checkpointing ---
        if self.rank == 0 and self.config.checkpoint_dir:
          # save best
          if self.config.save_best and val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(self.config.checkpoint_dir, self.config.best_ckpt_name)
            torch.save(self._unwrap(self.model).state_dict(), best_path)
          # save every epoch
          if self.config.save_every_epoch:
            ep_path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch+1:03d}.ckpt")
            torch.save(self._unwrap(self.model).state_dict(), ep_path)

      # save last
      if self.rank == 0 and self.config.checkpoint_dir:
        last_path = os.path.join(self.config.checkpoint_dir, self.config.last_ckpt_name)
        torch.save(self._unwrap(self.model).state_dict(), last_path)

    except KeyboardInterrupt:
      self.logger.info("Training interrupted by user")

    if self.config.distributed:
      self.dist_manager.cleanup()

    if self.config.use_wandb:
      self.metrics_logger.close()

    return {"history": history, "best_ckpt": best_path}

  def load_checkpoint(self, ckpt_path: str) -> None:
    """Load weights into the current model."""
    state = torch.load(ckpt_path, map_location=self.device)
    self._unwrap(self.model).load_state_dict(state)
    self.logger.info(f"Loaded checkpoint from {ckpt_path}")

  # --------- Internal helpers --------- #
  def _unwrap(self, model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

  def _setup_distributed_model(self) -> nn.Module:
    self.model = self.model.to(self.device)
    return DDP(self.model,
               device_ids=[self.config.gpu_ids[self.rank]],
               output_device=self.config.gpu_ids[self.rank])

  def _create_dataloader(self, dataset: Dataset, is_train: bool) -> DataLoader:
    if self.config.distributed:
      sampler = torch.utils.data.distributed.DistributedSampler(
          dataset,
          num_replicas=len(self.config.gpu_ids),
          rank=dist.get_rank(),
          shuffle=is_train)
    else:
      sampler = None

    return DataLoader(
        dataset,
        batch_size=self.config.batch_size,
        shuffle=(is_train and not self.config.distributed),
        sampler=sampler,
    )

  def _quantile_loss(self, pred: torch.Tensor, actual: torch.Tensor, quantile: float) -> torch.Tensor:
    dev = actual - pred
    loss_first = dev * quantile
    loss_second = -dev * (1.0 - quantile)
    return 2 * torch.where(loss_first >= 0, loss_first, loss_second)

  def _process_batch(self, batch: List[torch.Tensor]) -> tuple:
    x_context, x_padding, freq, x_future = [t.to(self.device, non_blocking=True) for t in batch]

    predictions = self.model(x_context, x_padding.float(), freq)
    predictions_mean = predictions[..., 0]                 # [B, N, horizon_len]
    last_patch_pred = predictions_mean[:, -1, :]           # [B, horizon_len]

    loss = self.loss_fn(last_patch_pred, x_future.squeeze(-1))
    if self.config.use_quantile_loss:
      quantiles = self.config.quantiles or create_quantiles()
      for i, quantile in enumerate(quantiles):
        last_patch_quantile = predictions[:, -1, :, i + 1]
        loss += torch.mean(self._quantile_loss(last_patch_quantile,
                                               x_future.squeeze(-1),
                                               quantile))
    return loss, predictions

  def _train_epoch(self, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   epoch: int) -> float:
    self.model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    iterator = train_loader
    if self.config.progress_bar and self.rank == 0:
      iterator = tqdm(train_loader,
                      desc=f"Train Epoch {epoch+1}/{self.config.num_epochs}",
                      leave=True)

    running = 0.0
    for step, batch in enumerate(iterator, 1):
      loss, _ = self._process_batch(batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      running += loss.item()

      if (step % self.config.log_every_n_steps == 0) and self.config.progress_bar and self.rank == 0:
        iterator.set_postfix(loss=running / self.config.log_every_n_steps)
        running = 0.0

    avg_loss = total_loss / num_batches

    if self.config.distributed:
      avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
      dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
      avg_loss = (avg_loss_tensor / dist.get_world_size()).item()

    return avg_loss

  def _validate(self, val_loader: DataLoader, epoch: int) -> float:
    self.model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    iterator = val_loader
    if self.config.progress_bar and self.rank == 0:
      iterator = tqdm(val_loader,
                      desc=f"Val   Epoch {epoch+1}/{self.config.num_epochs}",
                      leave=False)

    with torch.no_grad():
      for batch in iterator:
        loss, _ = self._process_batch(batch)
        total_loss += loss.item()

    avg_loss = total_loss / num_batches

    if self.config.distributed:
      avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
      dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
      avg_loss = (avg_loss_tensor / dist.get_world_size()).item()

    return avg_loss
