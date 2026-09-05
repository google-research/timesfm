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

"""Tests for the XReg helpers."""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("sklearn")

from timesfm.utils import xreg_lib


@pytest.mark.parametrize(
  ("shape", "expected_shape"),
  [
    ((0,), (0,)),
    ((3, 0), (4, 0)),
    ((0, 3), (0, 4)),
  ],
)
def test_to_padded_jax_array_preserves_empty_dimensions(shape, expected_shape):
  padded = xreg_lib._to_padded_jax_array(np.empty(shape))

  assert padded.shape == expected_shape


def test_to_padded_jax_array_still_pads_nonempty_dimensions():
  values = np.arange(6).reshape(3, 2)

  padded = xreg_lib._to_padded_jax_array(values)

  assert padded.shape == (4, 2)
  np.testing.assert_array_equal(np.asarray(padded)[:3], values)
  np.testing.assert_array_equal(np.asarray(padded)[3], np.zeros(2))


def test_fit_handles_single_category_without_intercept():
  model = xreg_lib.BatchedInContextXRegLinear(
    targets=[[1.0, 2.0, 3.0]],
    train_lens=[3],
    test_lens=[2],
    train_dynamic_categorical_covariates={"kind": [["same"] * 3]},
    test_dynamic_categorical_covariates={"kind": [["same"] * 2]},
  )

  outputs = model.fit(
    use_intercept=False,
    assert_covariates=True,
    assert_covariate_shapes=True,
  )

  np.testing.assert_array_equal(outputs, np.zeros((1, 2)))
