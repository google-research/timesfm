from types import SimpleNamespace

import numpy as np
import pytest
import torch

from timesfm import ForecastConfig
from timesfm.timesfm_2p5 import timesfm_2p5_torch


@pytest.mark.parametrize("magnitude", [1.0, 5e19])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_large_input_normalization(monkeypatch, magnitude, device):
  if device == "cuda" and not torch.cuda.is_available():
    pytest.skip("CUDA is not available")
  # Exercise the compiled-decode wrapper without downloading model weights.
  def decode(horizon, inputs, masks):
    torch.testing.assert_close(
        inputs.mean(dim=-1), torch.zeros(1, device=device), atol=1e-6, rtol=0,
    )
    torch.testing.assert_close(inputs.std(dim=-1), torch.ones(1, device=device))
    return torch.ones(1, 1, 128, 10, device=device), None, None

  monkeypatch.setattr(
      timesfm_2p5_torch, "TimesFM_2p5_200M_torch_module",
      lambda: SimpleNamespace(
          p=32, o=128, os=1024, q=10, device=device, device_count=1,
          config=SimpleNamespace(context_limit=16384), decode=decode,
      ),
  )
  model = timesfm_2p5_torch.TimesFM_2p5_200M_torch(torch_compile=False)
  model.compile(ForecastConfig(
      max_context=32, max_horizon=128, normalize_inputs=True,
      use_continuous_quantile_head=False, force_flip_invariance=False,
      infer_is_positive=False, fix_quantile_crossing=False,
  ))
  inputs = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), 8)[None] * magnitude
  _, forecast = model.compiled_decode(1, inputs, np.zeros_like(inputs, dtype=bool))
  reference = inputs.astype(np.float64)
  expected = reference.mean() + reference.std(ddof=1)
  np.testing.assert_allclose(forecast, expected, rtol=1e-5)
