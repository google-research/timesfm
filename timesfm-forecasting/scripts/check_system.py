#!/usr/bin/env python3
"""TimesFM System Requirements Preflight Checker.

MANDATORY: Run this script before loading TimesFM for the first time.
It checks RAM, GPU/VRAM, disk space, Python version, and package
installation so the agent never crashes a user's machine.

Usage:
    python check_system.py
    python check_system.py --model v2.5   # default
    python check_system.py --model v2.0   # archived 500M model
    python check_system.py --model v1.0   # archived 200M model
    python check_system.py --json         # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math


# ---------------------------------------------------------------------------
# Model requirement profiles
# ---------------------------------------------------------------------------

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "v2.5": {
        "name": "TimesFM 2.5 (200M)",
        "params": "200M",
        "min_ram_gb": 2.0,
        "recommended_ram_gb": 4.0,
        "min_vram_gb": 2.0,
        "recommended_vram_gb": 4.0,
        "disk_gb": 2.0,  # model weights + overhead
        "hf_repo": "google/timesfm-2.5-200m-pytorch",
    },
    "v2.0": {
        "name": "TimesFM 2.0 (500M)",
        "params": "500M",
        "min_ram_gb": 8.0,
        "recommended_ram_gb": 16.0,
        "min_vram_gb": 4.0,
        "recommended_vram_gb": 8.0,
        "disk_gb": 4.0,
        "hf_repo": "google/timesfm-2.0-500m-pytorch",
    },
    "v1.0": {
        "name": "TimesFM 1.0 (200M)",
        "params": "200M",
        "min_ram_gb": 4.0,
        "recommended_ram_gb": 8.0,
        "min_vram_gb": 2.0,
        "recommended_vram_gb": 4.0,
        "disk_gb": 2.0,
        "hf_repo": "google/timesfm-1.0-200m-pytorch",
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    status: str  # "pass", "warn", "fail"
    detail: str
    value: str = ""

    @property
    def icon(self) -> str:
        return {"pass": "✅", "warn": "⚠️", "fail": "🛑"}.get(self.status, "❓")

    def __str__(self) -> str:
        return f"[{self.name:<10}] {self.value:<40} {self.icon} {self.status.upper()}"


@dataclass
class SystemReport:
    model: str
    checks: list[CheckResult] = field(default_factory=list)
    verdict: str = ""
    verdict_detail: str = ""
    recommended_batch_size: int = 1
    mode: str = "cpu"  # "cpu", "gpu", "mps"

    @property
    def passed(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "passed": self.passed,
            "mode": self.mode,
            "recommended_batch_size": self.recommended_batch_size,
            "verdict": self.verdict,
            "verdict_detail": self.verdict_detail,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "value": c.value,
                }
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _get_total_ram_gb() -> float:
    """Return total physical RAM in GB, cross-platform."""
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / (1024 * 1024)
        elif sys.platform == "darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip()) / (1024**3)
        elif sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3)
    except Exception:
        pass

    # Fallback: use struct to estimate (unreliable)
    return struct.calcsize("P") * 8 / 8  # placeholder


def _get_available_ram_gb() -> float:
    """Return available RAM in GB."""
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) / (1024 * 1024)
        elif sys.platform == "darwin":
            import subprocess

            # Use vm_stat for available memory on macOS
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True
            )
            free = 0
            page_size = 4096
            for line in result.stdout.split("\n"):
                if "Pages free" in line or "Pages inactive" in line:
                    val = line.split(":")[1].strip().rstrip(".")
                    free += int(val) * page_size
            return free / (1024**3)
        elif sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024**3)
    except Exception:
        pass
    return 0.0


def check_ram(profile: dict[str, Any]) -> CheckResult:
    """Check if system has enough RAM."""
    total = _get_total_ram_gb()
    available = _get_available_ram_gb()
    min_ram = profile["min_ram_gb"]
    rec_ram = profile["recommended_ram_gb"]

    value = f"Total: {total:.1f} GB | Available: {available:.1f} GB"

    if total < min_ram:
        return CheckResult(
            name="RAM",
            status="fail",
            detail=(
                f"System has {total:.1f} GB RAM but {profile['name']} requires "
                f"at least {min_ram:.0f} GB. The model will likely fail to load "
                f"or cause the system to swap heavily and become unresponsive."
            ),
            value=value,
        )
    elif total < rec_ram:
        return CheckResult(
            name="RAM",
            status="warn",
            detail=(
                f"System has {total:.1f} GB RAM. {profile['name']} recommends "
                f"{rec_ram:.0f} GB. It may work with small batch sizes but could "
                f"be tight. Use per_core_batch_size=4 or lower."
            ),
            value=value,
        )
    else:
        return CheckResult(
            name="RAM",
            status="pass",
            detail=f"System has {total:.1f} GB RAM, meets {rec_ram:.0f} GB recommendation.",
            value=value,
        )


def check_gpu() -> CheckResult:
    """Check GPU availability and VRAM."""
    # Try CUDA first
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return CheckResult(
                name="GPU",
                status="pass",
                detail=f"{name} with {vram:.1f} GB VRAM detected.",
                value=f"{name} | VRAM: {vram:.1f} GB",
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return CheckResult(
                name="GPU",
                status="pass",
                detail="Apple Silicon MPS backend available. Uses unified memory.",
                value="Apple Silicon MPS",
            )
        else:
            return CheckResult(
                name="GPU",
                status="warn",
                detail=(
                    "No GPU detected. TimesFM will run on CPU (slower but functional). "
                    "Install CUDA-enabled PyTorch for GPU acceleration."
                ),
                value="None (CPU only)",
            )
    except ImportError:
        return CheckResult(
            name="GPU",
            status="warn",
            detail="PyTorch not installed — cannot check GPU. Install torch first.",
            value="Unknown (torch not installed)",
        )


def check_disk(profile: dict[str, Any]) -> CheckResult:
    """Check available disk space for model download."""
    # Check HuggingFace cache dir or home dir
    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = Path(hf_cache)
    check_dir = cache_dir if cache_dir.exists() else Path.home()

    usage = shutil.disk_usage(str(check_dir))
    free_gb = usage.free / (1024**3)
    required = profile["disk_gb"]

    value = f"Free: {free_gb:.1f} GB (in {check_dir})"

    if free_gb < required:
        return CheckResult(
            name="Disk",
            status="fail",
            detail=(
                f"Only {free_gb:.1f} GB free in {check_dir}. "
                f"Need at least {required:.0f} GB for model weights. "
                f"Free up space or set HF_HOME to a larger volume."
            ),
            value=value,
        )
    else:
        return CheckResult(
            name="Disk",
            status="pass",
            detail=f"{free_gb:.1f} GB available, exceeds {required:.0f} GB requirement.",
            value=value,
        )


def check_python() -> CheckResult:
    """Check Python version >= 3.10."""
    version = sys.version.split()[0]
    major, minor = sys.version_info[:2]

    if (major, minor) < (3, 10):
        return CheckResult(
            name="Python",
            status="fail",
            detail=f"Python {version} detected. TimesFM requires Python >= 3.10.",
            value=version,
        )
    else:
        return CheckResult(
            name="Python",
            status="pass",
            detail=f"Python {version} meets >= 3.10 requirement.",
            value=version,
        )


def check_package(pkg_name: str, import_name: str | None = None) -> CheckResult:
    """Check if a Python package is installed."""
    import_name = import_name or pkg_name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        return CheckResult(
            name=pkg_name,
            status="pass",
            detail=f"{pkg_name} {version} is installed.",
            value=f"Installed ({version})",
        )
    except ImportError:
        return CheckResult(
            name=pkg_name,
            status="warn",
            detail=f"{pkg_name} is not installed. Run: uv pip install {pkg_name}",
            value="Not installed",
        )


# ---------------------------------------------------------------------------
# Batch size recommendation
# ---------------------------------------------------------------------------


def recommend_batch_size(report: SystemReport) -> int:
    """Recommend per_core_batch_size based on available resources."""
    total_ram = _get_total_ram_gb()

    # Check if GPU is available
    gpu_check = next((c for c in report.checks if c.name == "GPU"), None)

    if gpu_check and gpu_check.status == "pass" and "VRAM" in gpu_check.value:
        # Extract VRAM
        try:
            vram_str = gpu_check.value.split("VRAM:")[1].strip().split()[0]
            vram = float(vram_str)
            if vram >= 24:
                return 256
            elif vram >= 16:
                return 128
            elif vram >= 8:
                return 64
            elif vram >= 4:
                return 32
            else:
                return 16
        except (ValueError, IndexError):
            return 32
    elif gpu_check and "MPS" in gpu_check.value:
        # Apple Silicon — use unified memory heuristic
        if total_ram >= 32:
            return 64
        elif total_ram >= 16:
            return 32
        else:
            return 16
    else:
        # CPU only
        if total_ram >= 32:
            return 64
        elif total_ram >= 16:
            return 32
        elif total_ram >= 8:
            return 8
        else:
            return 4


def estimate_memory_gb(
    num_series: int,
    context_length: int,
    horizon: int = 0,
    batch_size: int = 32,
    model_version: str = "v2.5",
) -> dict[str, float]:
    """Estimate memory requirements for a dataset.

    Args:
        num_series: Number of time series in the dataset
        context_length: Length of each time series context window
        horizon: Forecast horizon (optional, for output storage)
        batch_size: Batch size for inference
        model_version: Model version being used

    Returns:
        Dictionary with memory estimates in GB for different components
    """
    # Base model memory (weights + overhead)
    model_memory_gb = 0.8  # ~800MB for model weights
    overhead_gb = 0.5  # Python overhead, libraries, etc.

    # Input data memory: each value is float32 (4 bytes)
    # Formula: num_series * context_length * 4 bytes / (1024^3)
    input_gb = (num_series * context_length * 4) / (1024**3)

    # Batch processing memory (peak during inference)
    # Each batch needs: batch_size * context_length * 4 bytes
    batch_input_gb = (batch_size * context_length * 4) / (1024**3)

    # Output memory: horizon * num_series * quantiles * 4 bytes
    # Default is 10 quantiles (mean + 9 quantiles)
    num_quantiles = 10
    output_gb = (num_series * horizon * num_quantiles * 4) / (1024**3) if horizon > 0 else 0

    # Total memory with some headroom for intermediate computations
    total_gb = model_memory_gb + overhead_gb + input_gb + batch_input_gb + output_gb

    # Add 20% buffer for intermediate tensors and OS overhead
    total_with_buffer = total_gb * 1.2

    return {
        "model_weights": model_memory_gb,
        "overhead": overhead_gb,
        "input_data": input_gb,
        "batch_processing": batch_input_gb,
        "output_data": output_gb,
        "total": total_gb,
        "total_with_buffer": total_with_buffer,
    }


def check_dataset_fit(
    num_series: int,
    context_length: int,
    horizon: int = 0,
    batch_size: int = 32,
    model_version: str = "v2.5",
) -> tuple[bool, str, dict[str, float]]:
    """Check if a dataset will fit in available memory.

    Args:
        num_series: Number of time series in the dataset
        context_length: Length of each time series context window
        horizon: Forecast horizon (optional)
        batch_size: Batch size for inference
        model_version: Model version being used

    Returns:
        Tuple of (fits: bool, message: str, memory_details: dict)
    """
    memory = estimate_memory_gb(num_series, context_length, horizon, batch_size, model_version)
    total_ram = _get_total_ram_gb()
    available_ram = _get_available_ram_gb()

    required = memory["total_with_buffer"]

    # Leave 10% headroom for OS and other processes
    usable_ram = total_ram * 0.9
    usable_available = available_ram * 0.9 if available_ram > 0 else usable_ram

    if required > total_ram:
        return (
            False,
            f"Dataset requires {required:.1f} GB but system only has {total_ram:.1f} GB RAM. "
            f"Consider processing in chunks or using a machine with more RAM.",
            memory,
        )
    elif required > usable_available:
        return (
            False,
            f"Dataset requires {required:.1f} GB but only {available_ram:.1f} GB is available. "
            f"Close other applications or restart to free memory.",
            memory,
        )
    elif required > usable_ram * 0.8:
        return (
            True,
            f"Dataset will fit ({required:.1f} GB needed, {total_ram:.1f} GB total) "
            f"but memory usage will be high. Consider reducing batch_size.",
            memory,
        )
    else:
        return (
            True,
            f"Dataset fits comfortably: {required:.1f} GB needed, {total_ram:.1f} GB available.",
            memory,
        )


def print_memory_estimate(
    num_series: int,
    context_length: int,
    horizon: int = 0,
    batch_size: int = 32,
    model_version: str = "v2.5",
) -> None:
    """Print a detailed memory estimate for a dataset.

    Args:
        num_series: Number of time series in the dataset
        context_length: Length of each time series context window
        horizon: Forecast horizon (optional)
        batch_size: Batch size for inference
        model_version: Model version being used
    """
    memory = estimate_memory_gb(num_series, context_length, horizon, batch_size, model_version)
    total_ram = _get_total_ram_gb()
    available_ram = _get_available_ram_gb()

    print(f"\n{'=' * 50}")
    print(f" Memory Estimate for Dataset")
    print(f"{'=' * 50}")
    print(f"  Dataset: {num_series:,} series × {context_length} context length")
    if horizon > 0:
        print(f"  Horizon: {horizon} steps")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: {model_version}")
    print(f"{'-' * 50}")
    print(f"  Model weights:     {memory['model_weights']:.2f} GB")
    print(f"  Overhead:          {memory['overhead']:.2f} GB")
    print(f"  Input data:        {memory['input_data']:.2f} GB")
    print(f"  Batch processing:  {memory['batch_processing']:.2f} GB")
    if horizon > 0:
        print(f"  Output data:       {memory['output_data']:.2f} GB")
    print(f"{'-' * 50}")
    print(f"  Total (raw):       {memory['total']:.2f} GB")
    print(f"  Total (+20% buf):  {memory['total_with_buffer']:.2f} GB")
    print(f"{'-' * 50}")
    print(f"  System RAM:        {total_ram:.1f} GB")
    print(f"  Available RAM:     {available_ram:.1f} GB")
    print(f"{'=' * 50}")

    fits, message, _ = check_dataset_fit(
        num_series, context_length, horizon, batch_size, model_version
    )
    status_icon = "✅" if fits else "🛑"
    print(f"  {status_icon} {message}")
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_checks(model_version: str = "v2.5") -> SystemReport:
    """Run all system checks and return a report."""
    profile = MODEL_PROFILES[model_version]
    report = SystemReport(model=profile["name"])

    # Run checks
    report.checks.append(check_ram(profile))
    report.checks.append(check_gpu())
    report.checks.append(check_disk(profile))
    report.checks.append(check_python())
    report.checks.append(check_package("timesfm"))
    report.checks.append(check_package("torch"))

    # Determine mode
    gpu_check = next((c for c in report.checks if c.name == "GPU"), None)
    if gpu_check and gpu_check.status == "pass":
        if "MPS" in gpu_check.value:
            report.mode = "mps"
        else:
            report.mode = "gpu"
    else:
        report.mode = "cpu"

    # Batch size
    report.recommended_batch_size = recommend_batch_size(report)

    # Verdict
    if report.passed:
        report.verdict = (
            f"✅ System is ready for {profile['name']} ({report.mode.upper()} mode)"
        )
        report.verdict_detail = (
            f"Recommended: per_core_batch_size={report.recommended_batch_size}"
        )
    else:
        failed = [c for c in report.checks if c.status == "fail"]
        report.verdict = f"🛑 System does NOT meet requirements for {profile['name']}"
        report.verdict_detail = "; ".join(c.detail for c in failed)

    return report


def print_report(report: SystemReport) -> None:
    """Print a human-readable report to stdout."""
    print(f"\n{'=' * 50}")
    print(f"  TimesFM System Requirements Check")
    print(f"  Model: {report.model}")
    print(f"{'=' * 50}\n")

    for check in report.checks:
        print(f"  {check}")
    print()

    print(f"  VERDICT: {report.verdict}")
    if report.verdict_detail:
        print(f"  {report.verdict_detail}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check system requirements for TimesFM.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PROFILES.keys()),
        default="v2.5",
        help="Model version to check requirements for (default: v2.5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (machine-readable)",
    )
    # Dataset preflight options (NEW)
    dataset_group = parser.add_argument_group("dataset preflight (optional)")
    dataset_group.add_argument(
        "--num-series",
        type=int,
        metavar="N",
        help="Number of time series in your dataset (for memory estimation)",
    )
    dataset_group.add_argument(
        "--context-length",
        type=int,
        metavar="LEN",
        help="Length of each input time series (max_context value)",
    )
    dataset_group.add_argument(
        "--horizon",
        type=int,
        metavar="H",
        default=24,
        help="Forecast horizon length (default: 24)",
    )
    dataset_group.add_argument(
        "--batch-size",
        type=int,
        metavar="SIZE",
        default=32,
        help="per_core_batch_size from ForecastConfig (default: 32)",
    )
    dataset_group.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only show memory estimate, skip system checks",
    )
    args = parser.parse_args()

    # Handle dataset estimation only mode
    if args.estimate_only and args.num_series and args.context_length:
        print_memory_estimate(
            args.num_series,
            args.context_length,
            args.horizon,
            args.batch_size,
            args.model,
        )
        sys.exit(0)

    # Run system checks
    report = run_checks(args.model)

    # Add dataset check if parameters provided
    if args.num_series and args.context_length:
        print_memory_estimate(
            args.num_series,
            args.context_length,
            args.horizon,
            args.batch_size,
            args.model,
        )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)

    # Exit with non-zero if any check failed
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
