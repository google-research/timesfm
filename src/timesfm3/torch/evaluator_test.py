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

"""Tests for PyTorch TimesFM3Evaluator.

These tests cover the evaluator's own logic (empty input, univariate unrolling
and re-stacking, forwarding of the benchmark defaults, and variate chunking for
inputs above the per-forward variate limit) with the parent
``TimesFM3Forecaster.predict_batch`` mocked out, so no checkpoint is needed.
"""

import unittest
from unittest import mock

import numpy as np

from . import evaluator, timesfm3_forecaster

_HORIZON = 4
_NUM_QUANTILES = 9


def _make_evaluator():
  # __init__ would load a checkpoint. The code under test only calls
  # super().predict_batch, which is mocked, so it is not needed here.
  return evaluator.TimesFM3Evaluator.__new__(evaluator.TimesFM3Evaluator)


def _fake_outputs(n, with_quantiles=True):
  """Builds n parent outputs; output i is filled with the value i."""
  outs = []
  for i in range(n):
    quantiles = None
    if with_quantiles:
      quantiles = np.full((_HORIZON, _NUM_QUANTILES), float(i), dtype=np.float32)
    outs.append(
      timesfm3_forecaster.ForecastOutput(
        ts_id=None,
        forecast=np.full((_HORIZON,), float(i), dtype=np.float32),
        quantiles=quantiles,
      )
    )
  return outs


class TimesFM3EvaluatorUnivariateTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.evaluator = _make_evaluator()
    patcher = mock.patch.object(
      timesfm3_forecaster.TimesFM3Forecaster, "predict_batch"
    )
    self.parent = patcher.start()
    self.addCleanup(patcher.stop)

  def test_empty_contexts_yields_nothing(self):
    outs = list(self.evaluator.predict_batch([], horizon=_HORIZON))
    self.assertEqual(outs, [])
    self.parent.assert_not_called()

  def test_1d_contexts_use_benchmark_defaults(self):
    contexts = [np.arange(10, dtype=np.float32), np.arange(6, dtype=np.float32)]
    self.parent.return_value = _fake_outputs(2)

    outs = list(
      self.evaluator.predict_batch(contexts, horizon=_HORIZON, univariate=True)
    )

    self.parent.assert_called_once()
    kwargs = self.parent.call_args.kwargs
    self.assertEqual(kwargs["horizon"], _HORIZON)
    self.assertEqual(len(kwargs["contexts"]), 2)
    np.testing.assert_array_equal(kwargs["contexts"][0], contexts[0])
    np.testing.assert_array_equal(kwargs["contexts"][1], contexts[1])
    self.assertIsNone(kwargs["past_only_covariates"])
    self.assertIsNone(kwargs["past_future_covariates"])
    self.assertTrue(kwargs["return_quantiles"])
    self.assertTrue(kwargs["use_symmetric_averaging"])
    self.assertTrue(kwargs["make_positive"])
    self.assertTrue(kwargs["sort_quantiles"])
    self.assertFalse(kwargs["use_znorm"])
    self.assertEqual(kwargs["padding_mode"], "none")

    # 1-D inputs come back as 1-D forecasts.
    self.assertEqual(len(outs), 2)
    self.assertEqual(outs[0].forecast.shape, (_HORIZON,))
    self.assertEqual(outs[0].quantiles.shape, (_HORIZON, _NUM_QUANTILES))

  def test_overrides_are_forwarded(self):
    self.parent.return_value = _fake_outputs(1)

    list(
      self.evaluator.predict_batch(
        [np.zeros(10, dtype=np.float32)],
        horizon=_HORIZON,
        return_quantiles=False,
        use_symmetric_averaging=False,
        make_positive=False,
        sort_quantiles=False,
        use_znorm=True,
        padding_mode="edge",
        univariate=True,
      )
    )

    kwargs = self.parent.call_args.kwargs
    self.assertFalse(kwargs["return_quantiles"])
    self.assertFalse(kwargs["use_symmetric_averaging"])
    self.assertFalse(kwargs["make_positive"])
    self.assertFalse(kwargs["sort_quantiles"])
    self.assertTrue(kwargs["use_znorm"])
    self.assertEqual(kwargs["padding_mode"], "edge")

  def test_2d_context_is_unrolled_and_restacked(self):
    context = np.stack(
      [np.full(10, 1.0), np.full(10, 2.0), np.full(10, 3.0)]
    ).astype(np.float32)
    self.parent.return_value = _fake_outputs(3)

    outs = list(
      self.evaluator.predict_batch([context], horizon=_HORIZON, univariate=True)
    )

    flat = self.parent.call_args.kwargs["contexts"]
    self.assertEqual(len(flat), 3)
    for i, series in enumerate(flat):
      self.assertEqual(series.shape, (10,))
      np.testing.assert_array_equal(series, context[i])

    self.assertEqual(len(outs), 1)
    self.assertEqual(outs[0].forecast.shape, (3, _HORIZON))
    self.assertEqual(outs[0].quantiles.shape, (3, _HORIZON, _NUM_QUANTILES))
    # Variate order must be preserved when re-stacking.
    np.testing.assert_array_equal(outs[0].forecast[:, 0], [0.0, 1.0, 2.0])

  def test_mixed_1d_and_2d_contexts_keep_their_shapes(self):
    contexts = [
      np.zeros((2, 10), dtype=np.float32),
      np.zeros(10, dtype=np.float32),
      np.zeros((2, 10), dtype=np.float32),
    ]
    # Parent output i is filled with the value i, so each series' values show
    # which flattened outputs it was given.
    self.parent.return_value = _fake_outputs(5)

    outs = list(
      self.evaluator.predict_batch(contexts, horizon=_HORIZON, univariate=True)
    )

    self.assertEqual(len(self.parent.call_args.kwargs["contexts"]), 5)
    self.assertEqual(len(outs), 3)
    self.assertEqual(outs[0].forecast.shape, (2, _HORIZON))
    self.assertEqual(outs[1].forecast.shape, (_HORIZON,))
    self.assertEqual(outs[2].forecast.shape, (2, _HORIZON))
    np.testing.assert_array_equal(outs[0].forecast[:, 0], [0.0, 1.0])
    np.testing.assert_array_equal(outs[1].forecast, np.full(_HORIZON, 2.0))
    np.testing.assert_array_equal(outs[2].forecast[:, 0], [3.0, 4.0])

  def test_ts_ids_are_preserved(self):
    self.parent.return_value = _fake_outputs(2)

    outs = list(
      self.evaluator.predict_batch(
        [np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32)],
        horizon=_HORIZON,
        ts_ids=["a", "b"],
        univariate=True,
      )
    )

    self.assertEqual([o.ts_id for o in outs], ["a", "b"])

  def test_without_quantiles_returns_none(self):
    self.parent.return_value = _fake_outputs(1, with_quantiles=False)

    outs = list(
      self.evaluator.predict_batch(
        [np.zeros(10, dtype=np.float32)],
        horizon=_HORIZON,
        return_quantiles=False,
        univariate=True,
      )
    )

    self.assertIsNone(outs[0].quantiles)
    self.assertEqual(outs[0].forecast.shape, (_HORIZON,))


_CONTEXT_LEN = 16


def _rows(num_rows, length, offset=0.0):
  """Returns a (num_rows, length) array whose row i is filled with offset + i."""
  values = offset + np.arange(num_rows, dtype=np.float32)
  return values[:, None] * np.ones((1, length), dtype=np.float32)


def _echo_parent(**kwargs):
  """Stands in for the parent's predict_batch by echoing each row's last value.

  For a context of shape (variates, time) it returns a forecast of shape
  (variates, horizon) whose rows repeat that variate's last observed value, so
  the output values reveal which input row produced which output row.
  """
  outs = []
  for ctx in kwargs["contexts"]:
    ctx = np.asarray(ctx)
    forecast = np.repeat(ctx[:, -1:], _HORIZON, axis=1)
    quantiles = np.repeat(forecast[..., None], _NUM_QUANTILES, axis=2)
    outs.append(
      timesfm3_forecaster.ForecastOutput(
        ts_id=None, forecast=forecast, quantiles=quantiles
      )
    )
  return outs


class TimesFM3EvaluatorChunkingTest(unittest.TestCase):
  """Tests the variate-chunking path used when a batch exceeds the variate limit."""

  def setUp(self):
    super().setUp()
    self.evaluator = _make_evaluator()
    patcher = mock.patch.object(
      timesfm3_forecaster.TimesFM3Forecaster, "predict_batch"
    )
    self.parent = patcher.start()
    self.parent.side_effect = _echo_parent
    self.addCleanup(patcher.stop)

  def _assert_within_variate_limit(self):
    """Every parent call must keep targets plus covariates within the limit."""
    self.assertGreater(self.parent.call_count, 0)
    for call in self.parent.call_args_list:
      kwargs = call.kwargs
      for i, ctx in enumerate(kwargs["contexts"]):
        total = ctx.shape[0]
        for name in ("past_only_covariates", "past_future_covariates"):
          covs = kwargs[name]
          if covs is not None and covs[i] is not None:
            total += covs[i].shape[0]
        self.assertLessEqual(total, evaluator._MAX_VARIATES_PER_FORWARD)

  def test_small_inputs_are_passed_through_unchanged(self):
    contexts = [_rows(3, _CONTEXT_LEN), _rows(2, _CONTEXT_LEN)]
    past_only = [_rows(1, _CONTEXT_LEN), None]

    outs = list(
      self.evaluator.predict_batch(
        contexts,
        horizon=_HORIZON,
        past_only_covariates=past_only,
        ts_ids=["a", "b"],
      )
    )

    self.parent.assert_called_once()
    kwargs = self.parent.call_args.kwargs
    self.assertIs(kwargs["contexts"], contexts)
    self.assertIs(kwargs["past_only_covariates"], past_only)
    self.assertEqual(kwargs["ts_ids"], ["a", "b"])
    self.assertEqual(
      [o.forecast.shape for o in outs], [(3, _HORIZON), (2, _HORIZON)]
    )

  def test_targets_are_split_into_chunks_and_padding_is_trimmed(self):
    context = _rows(40, _CONTEXT_LEN)

    outs = list(self.evaluator.predict_batch([context], horizon=_HORIZON))

    self.assertEqual(self.parent.call_count, 2)
    first, second = (c.kwargs["contexts"][0] for c in self.parent.call_args_list)
    self.assertEqual(first.shape, (32, _CONTEXT_LEN))
    np.testing.assert_array_equal(first, context[:32])
    # The last chunk is padded up to a full 32 variates...
    self.assertEqual(second.shape, (32, _CONTEXT_LEN))
    np.testing.assert_array_equal(second[:8], context[32:])
    # ...but the padding must not leak into the results.
    self.assertEqual(len(outs), 1)
    self.assertEqual(outs[0].forecast.shape, (40, _HORIZON))
    self.assertEqual(outs[0].quantiles.shape, (40, _HORIZON, _NUM_QUANTILES))
    np.testing.assert_array_equal(outs[0].forecast[:, 0], np.arange(40))
    self._assert_within_variate_limit()

  def test_covariate_slots_are_reserved_before_chunking_targets(self):
    target = _rows(30, _CONTEXT_LEN)
    past_only = _rows(4, _CONTEXT_LEN, offset=100.0)
    past_future = _rows(5, _CONTEXT_LEN + _HORIZON, offset=200.0)

    outs = list(
      self.evaluator.predict_batch(
        [target],
        horizon=_HORIZON,
        past_only_covariates=[past_only],
        past_future_covariates=[past_future],
      )
    )

    # 32 - 4 - 5 = 23 target slots per pass, so 30 targets need two passes.
    self.assertEqual(self.parent.call_count, 2)
    for call in self.parent.call_args_list:
      kwargs = call.kwargs
      self.assertEqual(kwargs["contexts"][0].shape[0], 23)
      np.testing.assert_array_equal(kwargs["past_only_covariates"][0], past_only)
      np.testing.assert_array_equal(
        kwargs["past_future_covariates"][0], past_future
      )
    self.assertEqual(outs[0].forecast.shape, (30, _HORIZON))
    np.testing.assert_array_equal(outs[0].forecast[:, 0], np.arange(30))
    self._assert_within_variate_limit()

  def test_excess_future_covariates_are_subsampled_deterministically(self):
    target = _rows(2, _CONTEXT_LEN)
    past_only = _rows(3, _CONTEXT_LEN, offset=100.0)
    past_future = _rows(40, _CONTEXT_LEN + _HORIZON)

    def run():
      self.parent.reset_mock()
      list(
        self.evaluator.predict_batch(
          [target],
          horizon=_HORIZON,
          past_only_covariates=[past_only],
          past_future_covariates=[past_future],
        )
      )
      return self.parent.call_args_list[0].kwargs

    first = run()
    second = run()

    kept = first["past_future_covariates"][0]
    self.assertEqual(kept.shape, (31, _CONTEXT_LEN + _HORIZON))
    kept_ids = kept[:, 0]
    self.assertTrue(np.all(np.diff(kept_ids) > 0))  # sorted, no duplicates
    self.assertTrue(set(kept_ids.tolist()) <= set(range(40)))
    np.testing.assert_array_equal(kept, second["past_future_covariates"][0])
    self._assert_within_variate_limit()

  def test_1d_target_with_many_covariates_returns_1d_forecast(self):
    target = np.ones(_CONTEXT_LEN, dtype=np.float32)
    past_future = _rows(40, _CONTEXT_LEN + _HORIZON)

    outs = list(
      self.evaluator.predict_batch(
        [target], horizon=_HORIZON, past_future_covariates=[past_future]
      )
    )

    self.assertEqual(len(outs), 1)
    self.assertEqual(outs[0].forecast.shape, (_HORIZON,))
    self.assertEqual(outs[0].quantiles.shape, (_HORIZON, _NUM_QUANTILES))
    self._assert_within_variate_limit()

  def test_multiple_series_keep_order_and_ts_ids(self):
    series_a = _rows(40, _CONTEXT_LEN)
    series_b = _rows(40, _CONTEXT_LEN, offset=100.0)

    outs = list(
      self.evaluator.predict_batch(
        [series_a, series_b], horizon=_HORIZON, ts_ids=["a", "b"]
      )
    )

    self.assertEqual([o.ts_id for o in outs], ["a", "b"])
    for call in self.parent.call_args_list:
      self.assertEqual(call.kwargs["ts_ids"], ["a", "b"])
    np.testing.assert_array_equal(outs[0].forecast[:, 0], np.arange(40))
    np.testing.assert_array_equal(outs[1].forecast[:, 0], 100.0 + np.arange(40))


if __name__ == "__main__":
  unittest.main()