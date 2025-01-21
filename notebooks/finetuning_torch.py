"""
TimesFM Finetuner: A flexible framework for finetuning TimesFM models on custom datasets.

Example usage:
    ```python
    # Prepare datasets
    train_dataset = TimeSeriesDataset(train_data, context_length=128, horizon_length=32)
    val_dataset = TimeSeriesDataset(val_data, context_length=128, horizon_length=32)
    
    # Initialize model and configuration
    model = TimesFm(...)
    config = FinetuningConfig(
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-4,
        use_wandb=True
    )
    
    # Create finetuner
    finetuner = TimesFMFinetuner(model, config)
    
    # Finetune model
    results = finetuner.finetune(train_dataset, val_dataset)
    ```
"""

import abc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import multiprocessing as mp

from timesfm import TimesFm


@dataclass
class FinetuningConfig:
    """Configuration for TimesFM finetuning process."""

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Hardware parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    distributed: bool = False
    world_size: int = 1

    # Logging parameters
    use_wandb: bool = False
    wandb_project: str = "timesfm-finetuning"


class TimesFMFinetuner:
    """Main class for finetuning TimesFM models."""

    def __init__(
        self,
        model: TimesFm,
        config: FinetuningConfig,
        loss_fn: Optional[callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize TimesFM finetuner.

        Args:
            model: TimesFM model to finetune
            config: Finetuning configuration
            logger: Optional logger instance
        """
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.device = torch.device(config.device)
        self.loss_fn = loss_fn or (lambda x, y: torch.mean((x - y.squeeze(-1)) ** 2))  # MSELoss()

        if config.use_wandb:
            self._setup_wandb()

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        wandb.init(project=self.config.wandb_project, entity=self.config.wandb_entity, config=self.config.__dict__)

    def _create_dataloader(self, dataset: Dataset, name: str) -> DataLoader:
        """Create a dataloader from a dataset."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=name == "train",
            num_workers=mp.cpu_count(),
            pin_memory=self.device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
        )

    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(train_loader)

        for batch in train_loader:
            x_context, x_padding, freq, x_future = [t.to(self.device, non_blocking=True) for t in batch]

            predictions = self.model(x_context, x_padding.float(), freq)
            predictions_mean = predictions[..., 0]
            last_patch_pred = predictions_mean[:, -1, :]
            loss = self.loss_fn(last_patch_pred, x_future.squeeze(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / n_batches

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Perform validation."""
        self.model.eval()
        total_loss = 0.0

        for batch in val_loader:
            x_context, x_padding, freq, x_future = [t.to(self.device) for t in batch]

            predictions = self.model(x_context, x_padding.float(), freq)
            predictions_mean = predictions[..., 0]
            last_patch_pred = predictions_mean[:, -1, :]

            loss = self.loss_fn(last_patch_pred, x_future.squeeze(-1))
            total_loss += loss.item()

        return total_loss / len(val_loader)

    def finetune(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, Any]:
        """
        Finetune the TimesFM model on the provided datasets.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Dict containing training history and best model path
        """
        self.model = self.model.to(self.device)

        train_loader = self._create_dataloader(train_dataset, "train")
        val_loader = self._create_dataloader(val_dataset, "val")

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

        history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        self.logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

        try:
            for epoch in range(self.config.num_epochs):
                train_loss = self._train_epoch(train_loader, optimizer)

                val_loss = self._validate(val_loader)

                current_lr = optimizer.param_groups[0]["lr"]

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["learning_rate"].append(current_lr)

                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                }

                if self.config.use_wandb:
                    wandb.log(metrics)

                print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        return {"history": history}
