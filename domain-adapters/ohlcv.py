"""Let TimesFM 2.5 read the whole candle (high, low, volume), not just the close.

TimesFM is univariate: each 32-step patch of the (normalised) close series is
embedded by `model.model.input_ff_layer`. We add a small covariate encoder whose
output is summed into that embedding, per patch:

    token = input_ff_layer(close patch)  +  CovEncoder(features of the same 32 steps)

Per-step features (unit-free, so BTC and DOGE are comparable):
    up   = log(high / close)   / typical 1-step move in the window
    down = log(close / low)    / typical 1-step move in the window
    vol  = log(volume / mean volume in the window), clipped

The encoder's last layer starts at zero, so a fresh model is *exactly* the model
it was built on, and training can only add information.

TimesFM also runs a mirrored copy of the series (price -> -price) and averages the
two ("flip invariance"). In the mirror, highs become lows, so on that second pass
we swap the up/down features.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

N_FEATURES = 3
COLUMNS = ["close", "high", "low", "volume"]  # channel order used everywhere


def candle_features(ohlcv: torch.Tensor) -> torch.Tensor:
  """ohlcv: [B, T, 4] (close, high, low, volume) -> features [B, T, 3]."""
  close, high, low, vol = ohlcv.float().unbind(-1)
  logc = torch.log(close)
  step = (logc[:, 1:] - logc[:, :-1]).abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
  up = (torch.log(high) - logc).clamp_min(0) / step
  down = (logc - torch.log(low)).clamp_min(0) / step
  v = torch.log((vol + 1e-12) / (vol.mean(dim=1, keepdim=True) + 1e-12)).clamp(-5, 5)
  return torch.stack([up.clamp(0, 20), down.clamp(0, 20), v], dim=-1)


class CovEncoder(nn.Module):
  def __init__(self, patch_len: int = 32, hidden: int = 1280, width: int = 256):
    super().__init__()
    self.patch_len = patch_len
    self.net = nn.Sequential(
        nn.Linear(patch_len * N_FEATURES, width), nn.SiLU(), nn.Linear(width, hidden)
    )
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, feats: torch.Tensor) -> torch.Tensor:
    b, t, f = feats.shape
    return self.net(feats.reshape(b, t // self.patch_len, self.patch_len * f))


class CandleAware:
  """Attaches a CovEncoder to a (possibly PEFT-wrapped) TimesFM 2.5 model.

  Usage:
      ca = CandleAware.attach(model, device)
      with ca.features(ohlcv_context):     # [B, T, 4]
          out = model(past_values=ohlcv_context[..., 0], ...)
  Outside the `with` block the model behaves exactly like it did before.
  """

  def __init__(self, encoder: CovEncoder):
    self.encoder = encoder
    self._feats: torch.Tensor | None = None
    self._calls = 0
    self.enabled = True

  @classmethod
  def attach(cls, model, device, path: Path | None = None) -> "CandleAware":
    core = _core(model)
    enc = CovEncoder(core.config.patch_length, core.config.hidden_size).to(device)
    if path is not None:
      from safetensors.torch import load_file
      enc.load_state_dict(load_file(str(path)))
    ca = cls(enc)
    core.input_ff_layer.register_forward_hook(ca._hook)
    return ca

  def _hook(self, module, inputs, output):
    if self._feats is None or not self.enabled:
      return output
    feats = self._feats
    if self._calls % 2 == 1:  # mirrored pass: highs <-> lows
      feats = feats[..., [1, 0, 2]]
    self._calls += 1
    return output + self.encoder(feats.to(output.dtype)).to(output.dtype)

  @contextmanager
  def features(self, ohlcv: torch.Tensor):
    self._feats = candle_features(ohlcv)
    self._calls = 0
    try:
      yield
    finally:
      self._feats = None

  def save(self, path: Path) -> None:
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in self.encoder.state_dict().items()}, str(path))


def _core(model):
  """Find the TimesFm2_5Model inside a PEFT / ForPrediction wrapper."""
  for m in model.modules():
    if type(m).__name__ == "TimesFm2_5Model":
      return m
  raise ValueError("TimesFm2_5Model not found")


def load_horizon(adapter: str, device, base_model_id: str = "google/timesfm-2.5-200m-transformers"):
  """Load base + LoRA adapter (+ candle encoder if the adapter has one).
  Returns (model, candle_aware_or_None)."""
  from peft import PeftModel
  from transformers import TimesFm2_5ModelForPrediction

  model = TimesFm2_5ModelForPrediction.from_pretrained(base_model_id, dtype=torch.float32).to(device)
  if adapter == "none":
    return model.eval(), None
  path = Path(__file__).parent / "adapters" / adapter
  model = PeftModel.from_pretrained(model, path).eval()
  enc = path / "candle_encoder.safetensors"
  return model, (CandleAware.attach(model, device, enc) if enc.exists() else None)


def load_ohlcv(path: Path) -> tuple[np.ndarray, np.ndarray]:
  """Returns (times datetime64[ns] UTC-naive, values float32 [n, 4])."""
  import pandas as pd

  df = pd.read_parquet(path, columns=["timestamp"] + COLUMNS).dropna()
  df = df[(df["close"] > 0) & (df["low"] > 0)]
  times = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
  return times, df[COLUMNS].to_numpy(np.float32)
