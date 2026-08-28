"""Direct runner for executing time_bench_timesfm3.ipynb sequentially."""

import os
import sys
import json
import nbformat
from pathlib import Path

notebook_path = Path(__file__).parent / "time_bench_timesfm3.ipynb"
print(f"Executing TIME benchmark notebook: {notebook_path}", flush=True)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

global_ns = {"__name__": "__main__", "__file__": str(notebook_path)}

for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        code = "".join(cell.source)
        first_line = cell.source[0].strip() if cell.source else ""
        print(f"\n>>> Executing Notebook Cell {idx}: {first_line[:60]}...", flush=True)
        try:
            exec(code, global_ns)
            print(f">>> Cell {idx} completed successfully.", flush=True)
        except Exception as e:
            print(f">>> ERROR executing Cell {idx}: {e}", flush=True)
            raise e

print("\nTIME benchmark notebook execution fully completed!", flush=True)
