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

"""Tests for package import behavior."""

import os
import subprocess
import sys
from pathlib import Path


def test_import_does_not_configure_root_logger():
  source_dir = Path(__file__).parents[1] / "src"
  env = os.environ.copy()
  env["PYTHONPATH"] = str(source_dir)
  script = """
import io
import logging

stream = io.StringIO()
import timesfm
logging.basicConfig(stream=stream, level=logging.INFO)
logging.getLogger("timesfm-import-test").info("visible")
assert "visible" in stream.getvalue(), logging.root.handlers
"""

  subprocess.run(
    [sys.executable, "-c", script],
    check=True,
    env=env,
    timeout=120,
  )
