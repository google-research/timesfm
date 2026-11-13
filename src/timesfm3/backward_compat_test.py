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

"""The PyTorch backend moved into ``timesfm3.torch``; the old top-level import
paths (``from timesfm3.model import ...``) must keep working via thin shims.
These need PyTorch, so the module is skipped when it is not installed.
"""

import importlib
import unittest


def _torch_available() -> bool:
  try:
    import torch  # noqa: F401

    return True
  except ImportError:
    return False


@unittest.skipUnless(_torch_available(), "torch backend not installed")
class BackwardCompatImportTest(unittest.TestCase):
  """Legacy `timesfm3.<module>` import paths still resolve to the torch backend."""

  def test_legacy_named_imports(self):
    from timesfm3.timesfm3_forecaster import TimesFM3Forecaster
    from timesfm3.model import TimesFM3Torch
    from timesfm3.configs import TransformerConfig
    from timesfm3.evaluator import TimesFM3Evaluator

    from timesfm3.torch.timesfm3_forecaster import (
      TimesFM3Forecaster as _CanonForecaster,
    )
    from timesfm3.torch.model import TimesFM3Torch as _CanonTorch

    self.assertIs(TimesFM3Forecaster, _CanonForecaster)
    self.assertIs(TimesFM3Torch, _CanonTorch)
    self.assertTrue(TransformerConfig and TimesFM3Evaluator)

  def test_every_moved_module_is_importable(self):
    for name in (
      "configs",
      "cpm_revin_refine",
      "dense",
      "evaluator",
      "model",
      "normalization",
      "timesfm3_forecaster",
      "transformations",
      "transformer",
      "util",
    ):
      importlib.import_module(f"timesfm3.{name}")

  def test_legacy_private_names_still_resolve(self):
    # `_ModelConfig` is still listed in `timesfm3.__all__`, and these were all
    # importable from the old top-level modules before the torch move.
    from timesfm3.timesfm3_forecaster import _ModelConfig, _make_torch_model
    from timesfm3.evaluator import _MAX_VARIATES_PER_FORWARD
    from timesfm3.util import _make_safe_for_division

    from timesfm3.torch import evaluator, timesfm3_forecaster, util

    self.assertIs(_ModelConfig, timesfm3_forecaster._ModelConfig)
    self.assertIs(_make_torch_model, timesfm3_forecaster._make_torch_model)
    self.assertEqual(_MAX_VARIATES_PER_FORWARD, evaluator._MAX_VARIATES_PER_FORWARD)
    self.assertIs(_make_safe_for_division, util._make_safe_for_division)

  def test_patching_a_legacy_path_reaches_the_backend(self):
    from unittest import mock

    from timesfm3.torch import evaluator

    with mock.patch("timesfm3.evaluator._MAX_VARIATES_PER_FORWARD", 3):
      self.assertEqual(evaluator._MAX_VARIATES_PER_FORWARD, 3)


if __name__ == "__main__":
  unittest.main()
