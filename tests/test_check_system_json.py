# Copyright 2026 Google LLC
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

"""check_system.py --json must print only JSON, also with dataset options."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "timesfm-forecasting/scripts/check_system.py"
DATASET_ARGS = ["--num-series", "100", "--context-length", "512", "--horizon", "24"]


def run_json(*extra_args):
  completed = subprocess.run(
    [sys.executable, str(SCRIPT), "--json", *DATASET_ARGS, *extra_args],
    capture_output=True,
    text=True,
    encoding="utf-8",
  )
  return json.loads(completed.stdout)


def test_json_report_includes_dataset_estimate():
  result = run_json()

  assert result["model"] == "TimesFM 2.5 (200M)"
  assert isinstance(result["dataset"]["fits"], bool)
  assert result["dataset"]["memory_gb"]["total_with_buffer"] == pytest.approx(
    result["dataset"]["memory_gb"]["total"] * 1.2
  )


def test_json_estimate_only_prints_only_the_estimate():
  result = run_json("--estimate-only")

  assert set(result) == {"fits", "message", "memory_gb"}
