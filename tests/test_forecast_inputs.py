# Copyright 2025 Google LLC
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

"""Tests for input handling in ``TimesFM_2p5.forecast``.

``forecast`` pads the batch when the number of inputs is not a multiple of
the global batch size.  That padding must not be visible to the caller: a
caller that reuses its input list across calls would otherwise accumulate
padding entries and silently receive forecasts for them.

These tests drive the base class through a stub decode function so they
run without model weights.
"""

import numpy as np

from timesfm import configs
from timesfm.timesfm_2p5 import timesfm_2p5_base


class _StubTimesFM(timesfm_2p5_base.TimesFM_2p5):
  """Minimal concrete model whose decode step returns zeros."""

  def __init__(self, global_batch_size: int, horizon: int):
    self.global_batch_size = global_batch_size
    self.forecast_config = configs.ForecastConfig(
        max_context=64, max_horizon=horizon
    )
    self.calls = 0

    def _decode(h, values, masks):
      self.calls += 1
      b = len(values)
      return np.zeros((b, h)), np.zeros((b, h, 10))

    self.compiled_decode = _decode


def _series(n: int = 40) -> np.ndarray:
  return np.linspace(0.0, 1.0, n, dtype=np.float32)


class TestForecastDoesNotMutateInputs:

  def test_input_list_length_unchanged_when_padding_needed(self):
    model = _StubTimesFM(global_batch_size=4, horizon=8)
    inputs = [_series() for _ in range(3)]  # 3 % 4 != 0, so padding is added
    model.forecast(horizon=8, inputs=inputs)
    assert len(inputs) == 3

  def test_input_elements_unchanged(self):
    model = _StubTimesFM(global_batch_size=4, horizon=8)
    original = [_series() for _ in range(3)]
    inputs = list(original)
    model.forecast(horizon=8, inputs=inputs)
    assert [len(x) for x in inputs] == [len(x) for x in original]
    for a, b in zip(inputs, original):
      np.testing.assert_array_equal(a, b)

  def test_no_mutation_when_batch_divides_evenly(self):
    model = _StubTimesFM(global_batch_size=2, horizon=8)
    inputs = [_series() for _ in range(4)]
    model.forecast(horizon=8, inputs=inputs)
    assert len(inputs) == 4

  def test_repeated_calls_return_one_forecast_per_input(self):
    """Reusing the same list must not grow the returned batch."""
    model = _StubTimesFM(global_batch_size=4, horizon=8)
    inputs = [_series() for _ in range(3)]
    for _ in range(3):
      point, quantiles = model.forecast(horizon=8, inputs=inputs)
      assert point.shape == (3, 8)
      assert quantiles.shape == (3, 8, 10)
      assert len(inputs) == 3

  def test_accepts_a_tuple_of_inputs(self):
    """A non-list sequence must not raise when padding is required."""
    model = _StubTimesFM(global_batch_size=4, horizon=8)
    point, _ = model.forecast(horizon=8, inputs=tuple(_series() for _ in range(3)))
    assert point.shape == (3, 8)
