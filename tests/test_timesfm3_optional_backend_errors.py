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

"""Tests for the optional-backend contract of the ``timesfm3`` top-level re-exports.

``timesfm3`` re-exports the PyTorch backend lazily, so that importing it does
not require PyTorch. When PyTorch is missing, touching one of those names must
read as an absent attribute rather than raising out of the probe: ``hasattr``
and ``getattr(name, default)`` are how callers check for a backend before
choosing a fallback, and a crash at the check is what this pins down.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import timesfm3

_SRC_DIR = Path(timesfm3.__file__).resolve().parents[1]

_BLOCK_IMPORT_TEMPLATE = """
import sys


class _Blocker:
  \"\"\"Fail imports of `{module}` the way a missing dependency does.\"\"\"

  def find_spec(self, fullname, path=None, target=None):
    if fullname == "{module}" or fullname.startswith("{module}."):
      raise ModuleNotFoundError("No module named '{module}'")
    return None


sys.meta_path.insert(0, _Blocker())
"""

_TORCH_EXPORT_NAMES = (
  "ForecastOutput",
  "ModelConfig",
  "ResidualBlockConfig",
  "StackedTransformersConfig",
  "TimesFM3Evaluator",
  "TimesFM3Forecaster",
  "TimesFM3Torch",
  "TransformerConfig",
  "_ModelConfig",
)


def _run_python(
  snippet: str, blocked_modules: tuple[str, ...] = ()
) -> subprocess.CompletedProcess:
  """Run `snippet` in a fresh interpreter with optional imports blocked."""
  code = ""
  for module in blocked_modules:
    code += _BLOCK_IMPORT_TEMPLATE.format(module=module)
  code += textwrap.dedent(snippet)
  env = dict(os.environ)
  env["PYTHONPATH"] = os.pathsep.join(
    part for part in (str(_SRC_DIR), env.get("PYTHONPATH")) if part
  )
  return subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    env=env,
    check=False,
  )


def test_import_succeeds_without_the_torch_backend():
  result = _run_python(
    """
      import timesfm3

      assert timesfm3.__name__ == "timesfm3"
      """,
    blocked_modules=("torch",),
  )
  assert result.returncode == 0, result.stderr


def test_probes_report_a_missing_backend_as_unavailable():
  # Probing an optional backend must not raise: `hasattr` and `getattr` with
  # a default are how callers check for a backend before choosing a fallback,
  # so an unavailable backend has to read as absent rather than as an error.
  result = _run_python(
    f"""
      import timesfm3

      for name in {list(_TORCH_EXPORT_NAMES)!r}:
        assert hasattr(timesfm3, name) is False, name
        assert getattr(timesfm3, name, None) is None, name
      """,
    blocked_modules=("torch",),
  )
  assert result.returncode == 0, result.stderr


def test_direct_access_explains_how_to_install_the_backend():
  result = _run_python(
    """
      import timesfm3

      try:
        timesfm3.TimesFM3Forecaster
      except AttributeError as exc:
        message = str(exc)
        assert "TimesFM3Forecaster" in message, message
        assert 'timesfm[torch]' in message, message
        assert "No module named 'torch'" in message, message
        assert exc.__cause__ is not None, "original error must be chained"
      else:
        raise AssertionError("expected AttributeError")
      """,
    blocked_modules=("torch",),
  )
  assert result.returncode == 0, result.stderr


def test_from_import_raises_import_error_for_a_missing_backend():
  result = _run_python(
    """
      try:
        from timesfm3 import TimesFM3Forecaster
      except ImportError as exc:
        assert "TimesFM3Forecaster" in str(exc), str(exc)
      else:
        raise AssertionError("expected ImportError")
      """,
    blocked_modules=("torch",),
  )
  assert result.returncode == 0, result.stderr


def test_unknown_attribute_raises_standard_attribute_error():
  with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
    timesfm3.no_such_name  # noqa: B018 - the access itself is the assertion


@pytest.mark.skipif(
  importlib.util.find_spec("torch") is None,
  reason="the torch extra is not installed",
)
def test_torch_exports_are_available_when_installed():
  assert timesfm3.TimesFM3Forecaster is not None
  assert timesfm3.TimesFM3Torch is not None
