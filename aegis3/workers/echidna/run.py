#!/usr/bin/env python3
"""Echidna worker."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config = os.environ.get("ECHIDNA_CONFIG", "echidna.yaml")
    contract = os.environ.get("ECHIDNA_CONTRACT", "")
    cmd = ["echidna", "/src", "--config", f"/src/{config}", "--format", "json"]
    if contract:
        cmd += ["--contract", contract]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "result.json").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
