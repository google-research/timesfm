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
"""Tests for the MLX TimesFM3 reversible transformations.

MLX runs on Apple silicon, so the whole module is skipped when MLX is not installed.
"""
import unittest

import numpy as np

try:
  import mlx.core as mx

  from . import transformations as mlx_trans

  _HAS_MLX = True
except ImportError:
  _HAS_MLX = False


@unittest.skipUnless(_HAS_MLX, "mlx is not installed (Apple silicon only)")
class TransformationsTest(unittest.TestCase):
  """Tests reversible transformations."""

  def test_signed_log(self):
    x = mx.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=mx.float32)
    y = mlx_trans.signed_log(x)
    expected_y = mx.sign(x) * mx.log1p(mx.abs(x))
    np.testing.assert_allclose(np.array(y), np.array(expected_y), atol=1e-6)
    # Reverse round-trip
    z = mlx_trans.signed_log(y, reverse=True)
    np.testing.assert_allclose(np.array(z), np.array(x), atol=1e-5)

  def test_signed_log_derivatives_include_unit_slope_at_zero(self):
    # Regression test: signed_log used to be implemented as
    # sign(x) * f(abs(x)), which gives a dead (zero) gradient at x == 0
    # because mx.sign(0) == 0. The torch sibling deliberately avoids this
    # (see torch/transformations.py and torch/primitives_test.py's
    # test_signed_log_derivatives_include_unit_slope_at_zero), and MLX must
    # match: both branches should have unit slope at zero.
    for reverse in (False, True):
      with self.subTest(reverse=reverse):
        x = mx.array([-4.0, -0.1, -0.0, 0.0, 0.1, 4.0], dtype=mx.float32)

        def transform_sum(value):
          return mlx_trans.signed_log(value, reverse=reverse).sum()

        derivative = mx.grad(transform_sum)(x)
        expected = mx.exp(mx.abs(x)) if reverse else 1 / (1 + mx.abs(x))
        np.testing.assert_allclose(
            np.array(derivative), np.array(expected), atol=1e-6
        )

  def test_signed_log_values_registry_and_empty_inputs_remain_unchanged(self):
    for reverse in (False, True):
      with self.subTest(reverse=reverse):
        x = mx.array([-4.0, -1e-4, 0.0, 1e-4, 4.0], dtype=mx.float32)
        expected = mx.sign(x) * (
            mx.expm1(mx.abs(x)) if reverse else mx.log1p(mx.abs(x))
        )
        actual = mlx_trans.get_transform("signed_log")(x, reverse=reverse)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)
        empty = mx.zeros((2, 0), dtype=mx.float32)
        self.assertEqual(
            mlx_trans.signed_log(empty, reverse=reverse).shape, empty.shape
        )

  def test_signed_sqrt(self):
    x = mx.array([-9.0, -1.0, 0.0, 1.0, 16.0], dtype=mx.float32)
    y = mlx_trans.signed_sqrt(x)
    expected_y = mx.sign(x) * mx.sqrt(mx.abs(x))
    np.testing.assert_allclose(np.array(y), np.array(expected_y), atol=1e-6)
    # Reverse round-trip
    z = mlx_trans.signed_sqrt(y, reverse=True)
    np.testing.assert_allclose(np.array(z), np.array(x), atol=1e-5)


if __name__ == "__main__":
  unittest.main()
