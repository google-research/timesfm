# Copyright 2025 Google LLC
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

"""Multi-GPU PEFT trainer for TimesFM 2.5.

Supports:
* LoRA / DoRA adapters (via ``adapters.inject_adapters``)
* PyTorch DDP multi-GPU (``torchrun``)
* Mixed-precision training (fp16 / bf16)
* Gradient checkpointing
* Cosine-with-warmup LR schedule
* Early stopping & adapter-only checkpointing
* Optional W&B logging
"""

import logging
import math
import os
import time
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from .adapters import (
  inject_adapters,
    load_adapter_weights,
    merge_adapters,
    save_adapter_weights,
)
from .config import PEFTConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: access the raw model under potential DDP wrapper
# ---------------------------------------------------------------------------


def _unwrap(model: nn.Module) -> nn.Module:
  return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# Training forward — replicates the model's inference preprocessing so that
# gradients flow through the transformer + adapter parameters.
# ---------------------------------------------------------------------------


def _training_forward(
  model: nn.Module,
  context: torch.Tensor,
  masks: torch.Tensor,
  gradient_checkpointing: bool = False,
):
  """Run a differentiable forward pass for fine-tuning.

  This mirrors the pre-processing that ``TimesFM_2p5_200M_torch_module.decode``
  performs (patching → RevIN → transformer → output projections → un-RevIN),
  but without ``torch.no_grad()`` and without KV-cache / AR decoding.

  Args:
    model: The (possibly DDP-wrapped) model.
    context: ``(B, context_len)`` raw time-series values.
    masks: ``(B, context_len)`` boolean mask (``True`` = padding).
    gradient_checkpointing: Use activation checkpointing on transformer layers.

  Returns:
    ``(output_ts, output_qs)`` — *un-normalised* predictions, each of shape
    ``(B, N, output_patch_len, num_quantiles)``.
  """
  from timesfm.torch.util import revin, update_running_stats

  raw = _unwrap(model)
  B = context.shape[0]
  p = raw.p  # 32
  o = raw.o  # 128
  q = raw.q  # 10
  os_ = raw.os  # 1024

  # 1. Patch ----------------------------------------------------------------
  patched = context.reshape(B, -1, p)  # (B, N, 32)
  patched_masks = masks.reshape(B, -1, p)  # (B, N, 32)
  N = patched.shape[1]

  # 2. Running RevIN stats --------------------------------------------------
  n = torch.zeros(B, device=context.device)
  mu = torch.zeros(B, device=context.device)
  sigma = torch.zeros(B, device=context.device)
  patch_mus, patch_sigmas = [], []
  for i in range(N):
    (n, mu, sigma), _ = update_running_stats(
      n, mu, sigma, patched[:, i], patched_masks[:, i]
    )
    patch_mus.append(mu)
    patch_sigmas.append(sigma)
  ctx_mu = torch.stack(patch_mus, dim=1)  # (B, N)
  ctx_sigma = torch.stack(patch_sigmas, dim=1)  # (B, N)

  # 3. Normalise + mask -----------------------------------------------------
  normed = revin(patched, ctx_mu, ctx_sigma, reverse=False)
  normed = torch.where(patched_masks, 0.0, normed)

  # 4. Tokenise -------------------------------------------------------------
  tok_in = torch.cat([normed, patched_masks.to(normed.dtype)], dim=-1)
  embeddings = raw.tokenizer(tok_in)  # (B, N, model_dims)

  # 5. Transformer stack ----------------------------------------------------
  patch_mask = patched_masks[..., -1]  # (B, N) per-patch mask
  x = embeddings
  for layer in raw.stacked_xf:
    if gradient_checkpointing:
      x = torch.utils.checkpoint.checkpoint(
        _transformer_layer_fn, layer, x, patch_mask, use_reentrant=False
      )
    else:
      x, _ = layer(x, patch_mask)

  # 6. Output projections ---------------------------------------------------
  normed_ts = raw.output_projection_point(x)  # (B, N, o*q)
  normed_qs = raw.output_projection_quantiles(x)  # (B, N, os*q)

  # 7. Un-normalise ---------------------------------------------------------
  output_ts = revin(
    normed_ts.reshape(B, N, o, q), ctx_mu, ctx_sigma, reverse=True
  )
  output_qs = revin(
    normed_qs.reshape(B, N, os_, q), ctx_mu, ctx_sigma, reverse=True
  )

  return output_ts, output_qs


def _transformer_layer_fn(
  layer: nn.Module, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
  out, _ = layer(x, mask)
  return out


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

_DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _quantile_loss(
  pred: torch.Tensor, target: torch.Tensor, tau: float
) -> torch.Tensor:
  """Pinball (quantile) loss."""
  diff = target - pred
  return 2.0 * torch.where(diff >= 0, tau * diff, (tau - 1.0) * diff)


def _compute_loss(
  output_ts: torch.Tensor,
  target: torch.Tensor,
  horizon_len: int,
  use_quantile_loss: bool = False,
  quantile_loss_weight: float = 0.5,
):
  """Compute MSE (+ optional quantile) loss on the last-patch prediction.

  Args:
    output_ts: ``(B, N, 128, 10)`` denormalised forecast tensor.
    target: ``(B, horizon_len)`` ground-truth future values.
    horizon_len: Number of steps to compare.
    use_quantile_loss: Add pinball loss on quantile channels.
    quantile_loss_weight: Relative weight of the quantile term.

  Returns:
    Scalar loss tensor.
  """
  # Last input-patch → first horizon_len steps, median channel (idx 5).
  pred_median = output_ts[:, -1, :horizon_len, 5]  # (B, H)
  loss = torch.nn.functional.mse_loss(pred_median, target)

  if use_quantile_loss:
    q_loss = torch.tensor(0.0, device=loss.device)
    for qi, tau in enumerate(_DEFAULT_QUANTILES):
      pred_q = output_ts[:, -1, :horizon_len, qi + 1]  # channels 1-9
      q_loss = q_loss + _quantile_loss(pred_q, target, tau).mean()
    loss = loss + quantile_loss_weight * q_loss

  return loss


# ---------------------------------------------------------------------------
# PEFTTrainer
# ---------------------------------------------------------------------------


class PEFTTrainer:
  """Production-grade PEFT trainer for TimesFM 2.5 (PyTorch).

  Typical usage::

      from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
      model = TimesFM_2p5_200M_torch.from_pretrained(
          "google/timesfm-2.5-200m-pytorch", torch_compile=False
      )
      trainer = PEFTTrainer(model.model, PEFTConfig(...))
      history = trainer.fit(train_dataset, val_dataset)
      trainer.save_adapter("./adapter/adapter.safetensors")
  """

  def __init__(self, model: nn.Module, config: PEFTConfig):
    self.config = config
    self._setup_distributed()
    self._setup_seed(config.seed)

    # Inject adapters and freeze base weights.
    inject_adapters(model, config)

    # Move to device.
    self.device = torch.device(
      f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    model.to(self.device)

    self.raw_model = model
    if self.is_distributed:
      self.model = DDP(model, device_ids=[self.local_rank])
    else:
      self.model = model

    # Optimizer — only trainable (adapter) parameters.
    trainable = [p for p in model.parameters() if p.requires_grad]
    self.optimizer = torch.optim.AdamW(
      trainable,
      lr=config.learning_rate,
      weight_decay=config.weight_decay,
    )

    # AMP setup.
    self.autocast_dtype = {
      "fp16": torch.float16,
      "bf16": torch.bfloat16,
      "no": None,
    }[config.mixed_precision]
    self.scaler = (
      torch.amp.GradScaler("cuda")
      if config.mixed_precision == "fp16"
      else None
    )

    # Logging.
    self._wandb = None
    if config.use_wandb and self.is_main:
      try:
        import wandb

        wandb.init(project=config.wandb_project, config=config.__dict__)
        self._wandb = wandb
      except ImportError:
        logger.warning("wandb not installed — skipping W&B logging.")

    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    if self.is_main:
      logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{n_trainable:,}",
        f"{n_total:,}",
        100 * n_trainable / n_total,
      )

  # -- Distributed setup ---------------------------------------------------

  def _setup_distributed(self):
    self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    self.world_size = int(os.environ.get("WORLD_SIZE", 1))
    self.is_distributed = self.world_size > 1
    self.is_main = self.local_rank == 0

    if self.is_distributed and not dist.is_initialized():
      dist.init_process_group("nccl")
      torch.cuda.set_device(self.local_rank)

  @staticmethod
  def _setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

  # -- Data loaders --------------------------------------------------------

  def _make_loader(self, dataset: Dataset, is_train: bool) -> DataLoader:
    cfg = self.config
    sampler = None
    shuffle = is_train
    if self.is_distributed:
      sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=self.world_size,
        rank=self.local_rank,
        shuffle=is_train,
      )
      shuffle = False

    return DataLoader(
      dataset,
      batch_size=cfg.batch_size,
      shuffle=shuffle,
      sampler=sampler,
      num_workers=cfg.num_workers,
      pin_memory=True,
      drop_last=is_train,
    )

  # -- LR schedule ---------------------------------------------------------

  def _build_scheduler(self, total_steps: int):
    warmup_steps = int(self.config.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
      if step < warmup_steps:
        return step / max(1, warmup_steps)
      progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
      return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

  # -- Training / validation -----------------------------------------------

  def _train_step(self, batch):
    context, masks, target = [t.to(self.device, non_blocking=True) for t in batch]

    ctx_manager = (
      torch.amp.autocast("cuda", dtype=self.autocast_dtype)
      if self.autocast_dtype is not None
      else _nullcontext()
    )

    with ctx_manager:
      output_ts, _ = _training_forward(
        self.model,
        context,
        masks,
        gradient_checkpointing=self.config.gradient_checkpointing,
      )
      loss = _compute_loss(
        output_ts,
        target,
        self.config.horizon_len,
        use_quantile_loss=self.config.use_quantile_loss,
        quantile_loss_weight=self.config.quantile_loss_weight,
      )

    self.optimizer.zero_grad(set_to_none=True)
    if self.scaler is not None:
      self.scaler.scale(loss).backward()
      self.scaler.unscale_(self.optimizer)
      nn.utils.clip_grad_norm_(
        (p for p in self.raw_model.parameters() if p.requires_grad),
        self.config.gradient_clip_norm,
      )
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      loss.backward()
      nn.utils.clip_grad_norm_(
        (p for p in self.raw_model.parameters() if p.requires_grad),
        self.config.gradient_clip_norm,
      )
      self.optimizer.step()

    return loss.detach()

  @torch.no_grad()
  def _validate(self, val_loader: DataLoader) -> float:
    self.model.eval()
    total_loss = 0.0
    n = 0

    for batch in val_loader:
      context, masks, target = [t.to(self.device, non_blocking=True) for t in batch]

      ctx_manager = (
        torch.amp.autocast("cuda", dtype=self.autocast_dtype)
        if self.autocast_dtype is not None
        else _nullcontext()
      )
      with ctx_manager:
        output_ts, _ = _training_forward(
          self.model,
          context,
          masks,
          gradient_checkpointing=False,
        )
        loss = _compute_loss(
          output_ts,
          target,
          self.config.horizon_len,
          use_quantile_loss=self.config.use_quantile_loss,
          quantile_loss_weight=self.config.quantile_loss_weight,
        )
      total_loss += loss.item()
      n += 1

    avg = total_loss / max(n, 1)
    if self.is_distributed:
      t = torch.tensor(avg, device=self.device)
      dist.all_reduce(t, op=dist.ReduceOp.SUM)
      avg = (t / self.world_size).item()
    return avg

  # -- Main loop -----------------------------------------------------------

  def fit(
    self,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
  ) -> Dict[str, list]:
    """Run the full training loop.

    Args:
      train_dataset: Training data (``TimeSeriesDataset`` or any
        ``Dataset`` returning ``(context, mask, target)`` tensors).
      val_dataset: Optional validation data.

    Returns:
      Dictionary with ``train_loss``, ``val_loss``, ``lr`` histories.
    """
    cfg = self.config
    train_loader = self._make_loader(train_dataset, is_train=True)
    val_loader = (
      self._make_loader(val_dataset, is_train=False) if val_dataset else None
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg.num_epochs * steps_per_epoch
    scheduler = self._build_scheduler(total_steps)

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    if self.is_main:
      logger.info(
        "Training: %d epochs, %d steps/epoch, %d total steps",
        cfg.num_epochs,
        steps_per_epoch,
        total_steps,
      )

    for epoch in range(cfg.num_epochs):
      self.model.train()
      if self.is_distributed:
        train_loader.sampler.set_epoch(epoch)

      epoch_loss = 0.0
      t0 = time.time()

      for step, batch in enumerate(train_loader):
        loss = self._train_step(batch)
        scheduler.step()
        global_step += 1
        epoch_loss += loss.item()

        if self.is_main and global_step % cfg.log_every_n_steps == 0:
          lr = scheduler.get_last_lr()[0]
          logger.info(
            "[epoch %d  step %d/%d]  loss=%.5f  lr=%.2e",
            epoch + 1,
            step + 1,
            steps_per_epoch,
            loss.item(),
            lr,
          )
          if self._wandb is not None:
            self._wandb.log(
              {"train/loss": loss.item(), "train/lr": lr},
              step=global_step,
            )

      avg_train_loss = epoch_loss / max(steps_per_epoch, 1)
      history["train_loss"].append(avg_train_loss)
      history["lr"].append(scheduler.get_last_lr()[0])

      # Validation.
      val_loss = None
      if val_loader is not None:
        val_loss = self._validate(val_loader)
        history["val_loss"].append(val_loss)

      elapsed = time.time() - t0
      if self.is_main:
        msg = (
          f"[Epoch {epoch + 1}/{cfg.num_epochs}]  "
          f"train_loss={avg_train_loss:.5f}"
        )
        if val_loss is not None:
          msg += f"  val_loss={val_loss:.5f}"
        msg += f"  ({elapsed:.1f}s)"
        logger.info(msg)
        if self._wandb is not None:
          metrics = {"epoch": epoch + 1, "train/epoch_loss": avg_train_loss}
          if val_loss is not None:
            metrics["val/loss"] = val_loss
          self._wandb.log(metrics, step=global_step)

      # Checkpoint + early stopping.
      if val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        if self.is_main and cfg.save_every_n_epochs > 0:
          ckpt_path = os.path.join(cfg.checkpoint_dir, "best_adapter.safetensors")
          save_adapter_weights(self.raw_model, ckpt_path)
          logger.info("  ↳ Saved best adapter → %s", ckpt_path)
      elif val_loss is not None:
        patience_counter += 1
        if patience_counter >= cfg.early_stopping_patience:
          if self.is_main:
            logger.info("Early stopping triggered (patience=%d).", cfg.early_stopping_patience)
          break

      if (
        self.is_main
        and cfg.save_every_n_epochs > 0
        and (epoch + 1) % cfg.save_every_n_epochs == 0
      ):
        ep_path = os.path.join(
          cfg.checkpoint_dir, f"adapter_epoch{epoch + 1}.safetensors"
        )
        save_adapter_weights(self.raw_model, ep_path)

    # Cleanup.
    if self.is_distributed:
      dist.destroy_process_group()
    if self._wandb is not None:
      self._wandb.finish()

    return history

  # -- Convenience wrappers ------------------------------------------------

  def save_adapter(self, path: str) -> None:
    """Save adapter weights to *path* (safetensors format)."""
    save_adapter_weights(self.raw_model, path)

  def load_adapter(self, path: str) -> None:
    """Load adapter weights from *path*."""
    load_adapter_weights(self.raw_model, path)

  def merge_adapter(self) -> nn.Module:
    """Fold adapter weights into base model and return the raw model."""
    return merge_adapters(self.raw_model)


# ---------------------------------------------------------------------------
# Tiny helper to replace contextlib.nullcontext (available ≥3.7 but
# with async generics issues) for the AMP autocast conditional.
# ---------------------------------------------------------------------------

class _nullcontext:
  def __enter__(self):
    return None

  def __exit__(self, *_):
    return False
