from __future__ import annotations

import contextlib
import io
import os
import sys
import threading

import numpy as np


class TimesFMPredictor:
  """Lazy TimesFM runtime kept behind the forecast service interface."""

  def __init__(self):
    self.checkpoint = os.environ.get(
      "TIMESFM_CHECKPOINT", "google/timesfm-3.0-pytorch")
    self.local_only = os.environ.get("TIMESFM_LOCAL_ONLY", "1") == "1"
    self._model = None
    self._lock = threading.Lock()

  def _get_model(self):
    if self._model is not None:
      return self._model
    with self._lock:
      if self._model is not None:
        return self._model
      import torch
      from timesfm3 import ModelConfig, TimesFM3Evaluator

      device = "cuda" if torch.cuda.is_available() else "cpu"
      print(
        f"[timesfm-mcp] loading {self.checkpoint} on {device} ...",
        file=sys.stderr,
      )
      torch.set_float32_matmul_precision("high")
      with contextlib.redirect_stdout(io.StringIO()):
        self._model = TimesFM3Evaluator(ModelConfig(
          checkpoint_path=self.checkpoint,
          per_core_batch_size=4,
          device=device,
          local_files_only=self.local_only,
        ))
      print("[timesfm-mcp] model ready", file=sys.stderr)
      return self._model

  def predict(self, context: np.ndarray, horizon: int):
    model = self._get_model()
    with contextlib.redirect_stdout(io.StringIO()):
      output = list(model.predict_batch(
        contexts=[context.astype(np.float32)],
        horizon=horizon,
        return_quantiles=True,
        use_symmetric_averaging=False,
      ))[0]
    return output.forecast, output.quantiles[:, 0], output.quantiles[:, 8]

  def metadata(self):
    return {
      "name": "timesfm",
      "checkpoint": self.checkpoint,
      "local_files_only": self.local_only,
    }
