# Filename: tutorial_timesfm.py

import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import timesfm
from os import path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download


from timesfm.pytorch_patched_decoder import TimesFMConfig, PatchedTimeSeriesDecoder

import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# 1. Download stock data via yfinance
# --------------------------------------------------
def download_yfinance_data(ticker="AAPL", start="2020-01-01", end="2022-01-01"):
    """
    Download daily stock data for a given ticker from Yahoo Finance.
    Returns a pandas DataFrame with columns like 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    df = yf.download(ticker, start=start, end=end)
    df = df.dropna()
    return df["Close"].reset_index(drop=True)


# --------------------------------------------------
# 2. Create a dataset class for TimesFM
# --------------------------------------------------
class FinancialDataset(Dataset):
    def __init__(
        self,
        series: pd.Series,
        config: TimesFMConfig,
        context_length=128,  # how many past timesteps as input
        horizon_length=32,  # how many future steps to predict
    ):
        super().__init__()

        self.series = series.values.astype(np.float32)
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.config = config

        self.samples = []
        # We want to ensure we have at least context_length + horizon_length points.
        for start_idx in range(0, len(self.series) - (context_length + horizon_length)):
            end_idx = start_idx + context_length
            # context slice
            x_context = self.series[start_idx:end_idx]
            # future/horizon slice
            x_future = self.series[end_idx : end_idx + horizon_length]
            self.samples.append((x_context, x_future))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x_context, x_future = self.samples[index]
        # Convert to torch
        x_context = torch.tensor(x_context, dtype=torch.float32)
        x_future = torch.tensor(x_future, dtype=torch.float32)

        input_padding = torch.zeros_like(x_context)

        freq = torch.zeros(1, dtype=torch.long)

        return x_context, input_padding, freq, x_future


def collate_fn(batch):
    xs_context = [item[0] for item in batch]
    xs_padding = [item[1] for item in batch]
    freqs = [item[2] for item in batch]
    xs_future = [item[3] for item in batch]

    x_context = torch.stack(xs_context, dim=0)
    input_pad = torch.stack(xs_padding, dim=0)
    freq = torch.stack(freqs, dim=0)  # shape [B, 1]
    x_future = torch.stack(xs_future, dim=0)

    return x_context, input_pad, freq, x_future


def get_model(*, load_weights: bool = False):
    # standard model hack
    repo_id = "google/timesfm-2.0-500m-pytorch"
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cuda",
            per_core_batch_size=32,
            horizon_len=128,
            num_layers=50,
            use_positional_embedding=False,
            context_len=192,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=repo_id),
    )

    model = PatchedTimeSeriesDecoder(tfm._model_config)

    if load_weights:
        checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
        print(model.state_dict()["input_ff_layer.hidden_layer.0.weight"])
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(loaded_checkpoint)
        print("After loading:")
        print(model.state_dict()["input_ff_layer.hidden_layer.0.weight"])
    model = model.to(device)

    # import sys
    # sys.exit(-1)
    # repo_id = "google/timesfm-1.0-200m"
    return model, tfm._model_config


def train_model(
    ticker="AAPL", start="2015-01-01", end="2022-01-01", train_split=0.8, batch_size=8, num_epochs=20, pretrained=False
):
    df_close = download_yfinance_data(ticker, start=start, end=end)
    model, config = get_model(load_weights=pretrained)

    total_len = len(df_close)
    train_size = int(total_len * train_split)
    val_size = total_len - train_size

    train_series = df_close.iloc[:train_size].reset_index(drop=True)
    val_series = df_close.iloc[train_size:].reset_index(drop=True)

    train_dataset = FinancialDataset(
        series=train_series, config=config, context_length=128, horizon_length=config.horizon_len
    )
    val_dataset = FinancialDataset(
        series=val_series, config=config, context_length=128, horizon_length=config.horizon_len
    )
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x_context, x_padding, freq, x_future in train_dataloader:
            x_context, x_padding, freq, x_future = (
                x_context.to(device),
                x_padding.to(device),
                freq.to(device),
                x_future.to(device),
            )
            predictions = model(x_context, x_padding.float(), freq)
            # predictions shape => [B, N, horizon_len, (1 + #quantiles)]
            predictions_mean = predictions[..., 0]  # => [B, N, horizon_len]
            last_patch_pred = predictions_mean[:, -1, :]  # => [B, horizon_len]

            # x_future => [B, horizon_len]
            loss = torch.mean((last_patch_pred - x_future.squeeze(-1)) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # -------- Compute validation loss --------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_context, x_padding, freq, x_future in val_dataloader:
                x_context, x_padding, freq, x_future = (
                    x_context.to(device),
                    x_padding.to(device),
                    freq.to(device),
                    x_future.to(device),
                )
                predictions = model(x_context, x_padding.float(), freq)
                predictions_mean = predictions[..., 0]
                last_patch_pred = predictions_mean[:, -1, :]
                val_loss = torch.mean((last_patch_pred - x_future.squeeze(-1)) ** 2)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / max(len(val_dataloader), 1)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "timesfm_finetuned.pth")
    return model, train_dataloader, val_dataloader


def plot_predictions(model, dataloader):
    model.eval()
    with torch.no_grad():
        x_context, x_padding, freq, x_future = next(iter(dataloader))
        x_context, x_padding, freq, x_future = (
            x_context.to(device),
            x_padding.to(device),
            freq.to(device),
            x_future.to(device),
        )
        # Forward pass
        predictions = model(x_context, x_padding.float(), freq)
        # => [B, N, horizon_len, (1 + #quantiles)]
        predictions_mean = predictions[..., 0]  # => [B, N, horizon_len]
        last_patch_prediction = predictions_mean[:, -1, :]  # => [B, horizon_len]

        # We'll plot only the first sample in the batch
        i = 0
        pred_vals = last_patch_prediction[i].cpu().numpy()  # [horizon_len]
        context_vals = x_context[i].cpu().numpy()  # [context_len]
        future_vals = x_future[i].cpu().numpy()  # [horizon_len]

        horizon_len = future_vals.shape[0]
        context_len = context_vals.shape[0]

        plt.figure(figsize=(10, 5))

        # Plot context
        plt.plot(range(context_len), context_vals, label="Context (History)", color="blue")

        # Plot predicted future
        plt.plot(
            range(context_len, context_len + horizon_len),
            pred_vals,
            label="Predicted Future",
            color="orange",
        )

        # Plot ground truth future
        plt.plot(
            range(context_len, context_len + horizon_len),
            future_vals,
            label="Ground Truth Future",
            color="green",
            linestyle="--",
        )

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Model Forecast vs. Ground Truth")
        plt.legend()
        plt.show()
        plt.savefig("pic_predictions.png")


if __name__ == "__main__":
    # Example usage
    model, train_dl, val_dl = train_model(
        ticker="AAPL",
        start="2012-01-01",
        end="2019-01-01",
        train_split=0.8,
        batch_size=256,
        num_epochs=50,
        pretrained=True,
    )

    plot_predictions(model, val_dl)
