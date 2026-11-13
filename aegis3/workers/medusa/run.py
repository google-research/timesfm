#!/usr/bin/env python3
"""Medusa worker."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config = os.environ.get("MEDUSA_CONFIG", "medusa.json")
    cmd = ["medusa", "fuzz", "--config", f"/src/{config}"]
    proc = subprocess.run(cmd, cwd="/src", capture_output=True, text=True)
    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)
    # Medusa writes its corpus + reports under /src/crytic-export by default;
    # collect the most relevant artifact paths for downstream normalization.
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
