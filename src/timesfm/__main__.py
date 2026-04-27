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
"""Runtime hardware detection, install recommendation, and auto-install.

Usage:
    python -m timesfm                            # detect hardware, print recommendations
    python -m timesfm --install                  # detect and run the torch install for you
    python -m timesfm --install --backend flax   # install flax instead
    python -m timesfm --install --backend all    # install torch + flax + xreg
    python -m timesfm --install --yes            # skip confirmation prompt
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Hardware detection helpers
# ---------------------------------------------------------------------------

def _nvidia_smi_present() -> bool:
  """Returns True if nvidia-smi is on PATH (i.e. NVIDIA drivers are installed)."""
  return shutil.which("nvidia-smi") is not None


def _rocm_present() -> bool:
  """Returns True if rocm-smi is on PATH (i.e. AMD ROCm drivers are installed)."""
  return shutil.which("rocm-smi") is not None


def _tpu_present() -> bool:
  """Returns True if running on a Google Cloud TPU VM."""
  # GCE TPU VMs set this environment variable.
  return bool(os.environ.get("TPU_NAME") or os.environ.get("TPU_ACCELERATOR_TYPE"))


def _detect_torch() -> tuple[str | None, str]:
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


def _detect_jax() -> tuple[str | None, str]:
  try:
    import jax
    devices = jax.devices()
    if not devices:
      return "cpu", "no devices"
    d = devices[0]
    if d.platform == "tpu":
      return "tpu", f"{d.device_kind}"
    if d.platform == "gpu":
      return "cuda", f"{d.device_kind}"
    return "cpu", "CPU only"
  except ImportError:
    return None, "jax not installed"


def _cuda_torch_index(cuda_major: str) -> str:
  """Maps a CUDA major version string to the PyTorch wheel index URL."""
  table = {
    "12": "https://download.pytorch.org/whl/cu128",
    "11": "https://download.pytorch.org/whl/cu118",
  }
  return table.get(cuda_major, "https://download.pytorch.org/whl/cu128")


def _infer_backend(
  torch_backend: str | None,
  jax_backend: str | None,
  system: str,
  machine: str,
) -> str:
  """Infers the best backend from detected info, falling back to driver probes."""
  if torch_backend == "cuda" or jax_backend == "cuda":
    return "cuda"
  if torch_backend == "mps" or jax_backend == "mps":
    return "mps"
  if jax_backend == "tpu":
    return "tpu"
  # Neither framework installed yet - probe drivers directly.
  if _nvidia_smi_present():
    return "cuda"
  if _tpu_present():
    return "tpu"
  if _rocm_present():
    return "rocm"
  if system == "darwin" and machine in ("arm64", "aarch64"):
    return "mps"
  return "cpu"


# ---------------------------------------------------------------------------
# Install-command builders
# ---------------------------------------------------------------------------

def _installer() -> list[str]:
  """Returns the base install argv: uv pip install if available, else pip."""
  if shutil.which("uv"):
    return ["uv", "pip", "install"]
  return [sys.executable, "-m", "pip", "install"]


def _build_install_commands(backend: str) -> dict[str, list[str]]:
  """Returns {label: argv} for each installable component given a backend."""
  inst = _installer()
  pkg = "."  # works for both local editable install and PyPI users

  if backend == "cuda":
    cuda_major = "12"
    try:
      import torch
      cuda_ver = torch.version.cuda or ""
      cuda_major = cuda_ver.split(".")[0] if cuda_ver else "12"
    except ImportError:
      cuda_major = "12"
    index = _cuda_torch_index(cuda_major)
    return {
      "torch  (NVIDIA CUDA)": inst + [f"{pkg}[torch]", "--index-url", index],
      "flax   (NVIDIA CUDA 12)": inst + [f"{pkg}[flax-cuda]"],
      "xreg   (NVIDIA CUDA 12)": inst + [f"{pkg}[xreg-cuda]"],
    }

  elif backend == "mps":
    return {
      "torch  (Apple Silicon - MPS)": inst + [f"{pkg}[torch]"],
      "flax   (Apple Silicon - Metal)": inst + [f"{pkg}[flax-metal]"],
      "xreg   (CPU, recommended on macOS)": inst + [f"{pkg}[xreg-cpu]"],
    }

  elif backend == "tpu":
    return {
      "flax   (Google TPU)": inst + [f"{pkg}[flax-tpu]"],
      "xreg   (CPU)": inst + [f"{pkg}[xreg-cpu]"],
    }

  elif backend == "rocm":
    # PyTorch ROCm wheel index - update the version tag as ROCm releases progress.
    rocm_index = "https://download.pytorch.org/whl/rocm6.2"
    return {
      "torch  (AMD ROCm)": inst + [f"{pkg}[torch]", "--index-url", rocm_index],
      "flax   (CPU - JAX ROCm is experimental, install manually if needed)": inst + [f"{pkg}[flax-cpu]"],
      "xreg   (CPU)": inst + [f"{pkg}[xreg-cpu]"],
    }

  else:  # cpu
    return {
      "torch  (CPU)": inst + [f"{pkg}[torch]"],
      "flax   (CPU)": inst + [f"{pkg}[flax-cpu]"],
      "xreg   (CPU)": inst + [f"{pkg}[xreg-cpu]"],
    }


def _filter_by_backend_arg(
  cmds: dict[str, list[str]], backend_arg: str
) -> dict[str, list[str]]:
  if backend_arg == "all":
    return cmds
  return {k: v for k, v in cmds.items() if backend_arg in k.lower()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser(
    prog="python -m timesfm",
    description="Detect hardware and install the right timesfm extras.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
examples:
  python -m timesfm                          # show recommendations
  python -m timesfm --install                # auto-install torch (prompts once)
  python -m timesfm --install --yes          # auto-install torch, no prompt
  python -m timesfm --install --backend all  # install torch + flax + xreg
  python -m timesfm --install --backend flax # install flax only
    """,
  )
  parser.add_argument(
    "--install",
    action="store_true",
    help="Run the recommended install command(s) automatically.",
  )
  parser.add_argument(
    "--backend",
    choices=["torch", "flax", "xreg", "all"],
    default="torch",
    help="Which component(s) to install when --install is set (default: torch).",
  )
  parser.add_argument(
    "--yes", "-y",
    action="store_true",
    help="Skip the confirmation prompt when --install is used.",
  )
  args = parser.parse_args()

  machine = platform.machine().lower()
  system = sys.platform
  torch_backend, torch_detail = _detect_torch()
  jax_backend, jax_detail = _detect_jax()
  backend = _infer_backend(torch_backend, jax_backend, system, machine)

  print("timesfm hardware detection")
  print("=" * 40)
  print(f"OS       : {system} / {platform.machine()}")
  print(f"Python   : {sys.version.split()[0]}")
  print(f"Torch    : {torch_detail}")
  print(f"JAX      : {jax_detail}")
  print(f"Backend  : {backend}")
  print()

  install_cmds = _build_install_commands(backend)

  if not args.install:
    print("Recommended install commands:")
    print("-" * 40)
    for label, argv in install_cmds.items():
      print(f"  # {label}:")
      print(f"  {' '.join(argv)}")
      print()
    print("Run with --install to execute automatically:")
    print("  python -m timesfm --install")
    print("  python -m timesfm --install --backend all --yes")
    print()
    print("Full extra reference: https://github.com/google-research/timesfm#install")
    return

  # --- Install mode ---
  to_install = _filter_by_backend_arg(install_cmds, args.backend)
  if not to_install:
    print(f"No install commands matched --backend={args.backend}.")
    sys.exit(1)

  print("About to run:")
  for label, argv in to_install.items():
    print(f"  [{label}]")
    print(f"    {' '.join(argv)}")
  print()

  if not args.yes:
    try:
      answer = input("Proceed? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
      print("\nAborted.")
      sys.exit(0)
    if answer not in ("", "y", "yes"):
      print("Aborted.")
      sys.exit(0)

  for label, argv in to_install.items():
    print(f"\n--- Installing: {label} ---")
    result = subprocess.run(argv)
    if result.returncode != 0:
      print(f"\nInstall failed for '{label}' (exit code {result.returncode}).")
      sys.exit(result.returncode)

  print("\nAll installs completed.")
  print("Re-run `python -m timesfm` to verify the detected backends.")


if __name__ == "__main__":
  main()
