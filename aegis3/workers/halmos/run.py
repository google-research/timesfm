#!/usr/bin/env python3
"""Halmos worker."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["halmos", "--json-output", str(out_dir / "result.json")]
    match = os.environ.get("HALMOS_MATCH_CONTRACT")
    if match:
        cmd += ["--match-contract", match]
    proc = subprocess.run(cmd, cwd="/src", capture_output=True, text=True)
    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
