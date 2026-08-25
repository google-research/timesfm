#!/usr/bin/env python3
"""Slither worker entrypoint.

Inputs (env):
  OUT_DIR       — defaults to /out
Outputs:
  $OUT_DIR/result.sarif
  $OUT_DIR/result.json
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sarif = out_dir / "result.sarif"
    js = out_dir / "result.json"

    cmd = [
        "slither", "/src",
        "--sarif", str(sarif),
        "--json", str(js),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)
    # Slither returns non-zero when findings exist; that's expected.
    return 0 if (sarif.exists() or js.exists()) else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
