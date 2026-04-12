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
"""Runtime hardware detection and install recommendation.

Usage:
    python -m timesfm
"""

import platform
import sys


def _detect_torch():
  try:
    import torch
    if torch.cuda.is_available():
      name = torch.cuda.get_device_name(0)
      vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
      return "cuda", f"{name} ({vram:.1f} GB VRAM)"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      return "mps", "Apple Silicon (unified memory)"
    return "cpu", "no GPU detected"
  except ImportError:
    return None, "torch not installed"


def _detect_jax():
  try:
    import jax
    devices = jax.devices()
    if not devices:
      return "cpu", "no devices"
    d = devices[0]
    if d.platform == "gpu":
      return "cuda", f"{d.device_kind}"
    if d.platform == "tpu":
      return "tpu", f"{d.device_kind}"
    return "cpu", "CPU only"
  except ImportError:
    return None, "jax not installed"


def main():
  print("timesfm hardware detection\n" + "=" * 40)

  machine = platform.machine().lower()
  system = sys.platform

  # Detect via torch (most reliable for CUDA/MPS)
  torch_backend, torch_detail = _detect_torch()
  # Detect via jax if installed
  jax_backend, jax_detail = _detect_jax()

  # Determine actual backend
  backend = torch_backend or jax_backend or "cpu"

  print(f"OS       : {system} / {platform.machine()}")
  print(f"Python   : {sys.version.split()[0]}")
  print(f"Torch    : {torch_detail}")
  print(f"JAX      : {jax_detail}")
  print()

  # Recommendations
  print("Recommended install commands:")
  print("-" * 40)

  if backend == "cuda":
    # Try to get CUDA version
    cuda_ver = "unknown"
    try:
      import torch
      cuda_ver = torch.version.cuda or "unknown"
    except ImportError:
      pass
    # Map CUDA version to PyTorch index
    if cuda_ver.startswith("12"):
      torch_index = "https://download.pytorch.org/whl/cu124"
    elif cuda_ver.startswith("11"):
      torch_index = "https://download.pytorch.org/whl/cu118"
    else:
      torch_index = "https://download.pytorch.org/whl/cu124"

    print("  # PyTorch (NVIDIA GPU):")
    print(f"  pip install timesfm[torch] --index-url {torch_index}")
    print()
    print("  # Flax/JAX (NVIDIA GPU, CUDA 12):")
    print("  pip install timesfm[flax-cuda]")
    print()
    print("  # XReg covariates (NVIDIA GPU):")
    print("  pip install timesfm[xreg-cuda]")

  elif backend == "mps" or (system == "darwin" and machine in ("arm64", "aarch64")):
    print("  # PyTorch (Apple Silicon - MPS included by default):")
    print("  pip install timesfm[torch]")
    print()
    print("  # Flax/JAX (Apple Silicon - Metal, experimental):")
    print("  pip install timesfm[flax-metal]")
    print()
    print("  # XReg covariates (CPU recommended for stability):")
    print("  pip install timesfm[xreg-cpu]")
    print()
    print("  # Or let us auto-select for you:")
    print("  pip install timesfm[flax-auto]")

  else:
    print("  # PyTorch (CPU):")
    print("  pip install timesfm[torch]")
    print()
    print("  # Flax/JAX (CPU):")
    print("  pip install timesfm[flax-cpu]")
    print()
    print("  # XReg covariates (CPU):")
    print("  pip install timesfm[xreg-cpu]")
    print()
    print("  If you have an NVIDIA GPU, install CUDA and re-run this command.")

  print()
  print("Full extra reference: https://github.com/google-research/timesfm#install")


if __name__ == "__main__":
  main()
