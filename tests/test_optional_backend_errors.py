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

"""Tests for actionable errors when optional TimesFM backends are unavailable.

`import timesfm` must keep working without the optional extras (`timesfm[torch]`,
`timesfm[flax]`), and touching a backend whose import failed must report the
original failure plus the extra to install instead of a bare `AttributeError`
(see #318 and #287).

Missing dependencies are simulated with an import blocker inside subprocesses,
so these tests behave the same whether or not the extras are installed in the
running environment.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import timesfm

_SRC_DIR = Path(timesfm.__file__).resolve().parents[1]

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


def test_import_succeeds_without_any_optional_backend():
  result = _run_python(
    """
      import timesfm

      assert timesfm.ForecastConfig is not None
      """,
    blocked_modules=("torch", "jax"),
  )
  assert result.returncode == 0, result.stderr


def test_torch_backend_error_names_the_extra():
  result = _run_python(
    """
      import timesfm

      try:
        timesfm.TimesFM_2p5_200M_torch
      except ImportError as exc:
        message = str(exc)
        assert "TimesFM_2p5_200M_torch" in message, message
        assert 'timesfm[torch]' in message, message
        assert "No module named 'torch'" in message, message
        assert exc.__cause__ is not None, "original error must be chained"
      else:
        raise AssertionError("expected ImportError")
      """,
    blocked_modules=("torch",),
  )
  assert result.returncode == 0, result.stderr


def test_flax_backend_error_names_the_extra():
  result = _run_python(
    """
      import timesfm

      try:
        timesfm.TimesFM_2p5_200M_flax
      except ImportError as exc:
        message = str(exc)
        assert "TimesFM_2p5_200M_flax" in message, message
        assert 'timesfm[flax]' in message, message
        assert exc.__cause__ is not None, "original error must be chained"
      else:
        raise AssertionError("expected ImportError")
      """,
    blocked_modules=("jax",),
  )
  assert result.returncode == 0, result.stderr


def test_from_import_reports_the_missing_extra():
  result = _run_python(
    """
      try:
        from timesfm import TimesFM3Forecaster
      except ImportError as exc:
        assert 'timesfm[torch]' in str(exc), str(exc)
      else:
        raise AssertionError("expected ImportError")
      """,
    blocked_modules=("torch",),
  )
  assert result.returncode == 0, result.stderr


def test_unknown_attribute_raises_standard_attribute_error():
  with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
    timesfm.no_such_name  # noqa: B018 - the access itself is the assertion


@pytest.mark.skipif(
  importlib.util.find_spec("torch") is None,
  reason="the torch extra is not installed",
)
def test_torch_backend_is_available_when_installed():
  assert timesfm.TimesFM_2p5_200M_torch is not None
