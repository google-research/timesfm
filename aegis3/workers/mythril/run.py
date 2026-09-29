#!/usr/bin/env python3
"""Mythril worker."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = os.environ.get("MYTHRIL_TARGET", "/src")
    solv = os.environ.get("SOLC_VERSION", "0.8.24")
    cmd = ["myth", "analyze", target, "--solv", solv, "-o", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "result.json").write_text(proc.stdout or "{}")
    (out_dir / "stderr.log").write_text(proc.stderr)
    return 0  # mythril returns non-zero when issues exist; that's expected


if __name__ == "__main__":
    sys.exit(main())
