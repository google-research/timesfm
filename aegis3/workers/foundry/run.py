#!/usr/bin/env python3
"""Foundry worker — runs `forge test --json` and stores output."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = shlex.split(os.environ.get("FORGE_ARGS", ""))
    cmd = ["forge", "test", "--json", *extra]
    proc = subprocess.run(cmd, cwd="/src", capture_output=True, text=True)
    (out_dir / "stderr.log").write_text(proc.stderr)
    try:
        parsed = json.loads(proc.stdout or "{}")
        (out_dir / "result.json").write_text(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        (out_dir / "result.json").write_text(proc.stdout)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
