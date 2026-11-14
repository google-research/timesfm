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

"""Tests for PyTorch TimesFM3Forecaster."""

import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import configs, evaluator, timesfm3_forecaster
from . import model as torch_model_lib


class FakeModel(torch.nn.Module):
  """A tiny fake model for testing."""

  def __init__(self, input_patch_len=8, output_patch_len=8, quantiles=3):
    super().__init__()
    self.input_patch_len = input_patch_len
    self.output_patch_len = output_patch_len
    self.quantiles = quantiles

  def decode(
    self,
    target,
    autoregressive_index=0,
    horizon=8,
    past_only_covariates=None,
    past_future_covariates=None,
    past_only_mask=None,
    past_future_mask=None,
    mask=None,
  ):
    del autoregressive_index, past_only_covariates, past_future_covariates, mask
    del past_only_mask, past_future_mask
    b, v, _ = target.shape
    shape = (b, v, horizon, self.quantiles)
    return torch.ones(shape, dtype=torch.float32) * 2.0


class TimesFM3ForecasterTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.config = timesfm3_forecaster._ModelConfig(
      checkpoint_path="/fake/path",
      per_core_batch_size=2,
      input_patch_length=8,
      output_patch_length=8,
      median_quantile_index=1,
    )

  def test_linear_interpolation(self):
    arr = np.array([1.0, np.nan, 3.0])
    res = timesfm3_forecaster.linear_interpolation(arr)
    np.testing.assert_array_equal(res, np.array([1.0, 2.0, 3.0]))

  def test_linear_interpolation_all_nan(self):
    arr = np.array([np.nan, np.nan, np.nan])
    res = timesfm3_forecaster.linear_interpolation(arr)
    np.testing.assert_array_equal(res, np.array([0.0, 0.0, 0.0]))

  def test_znorm_stats(self):
    arr = np.array([1.0, 2.0, 3.0])
    mu, sigma = timesfm3_forecaster._znorm_stats(arr)
    self.assertAlmostEqual(mu, 2.0)
    self.assertAlmostEqual(sigma, np.std([1.0, 2.0, 3.0]))

  def test_is_nonnegative(self):
    arr_pos = np.array([0.0, 1.0, 2.0])
    arr_neg = np.array([-1.0, 1.0, 2.0])
    self.assertTrue(timesfm3_forecaster._is_nonnegative(arr_pos))
    self.assertFalse(timesfm3_forecaster._is_nonnegative(arr_neg))

  @mock.patch.object(
    timesfm3_forecaster.TimesFM3Forecaster, "_init_model", autospec=True
  )
  def test_univariate_predict(self, _):
    forecaster = timesfm3_forecaster.TimesFM3Forecaster(self.config)
    forecaster.model = FakeModel()
    forecaster.device = torch.device("cpu")

    ctx = np.array([1.0, 2.0, 3.0, 4.0])
    out = forecaster.predict(ctx, horizon=4)
    self.assertEqual(out.forecast.shape, (4,))
    np.testing.assert_array_equal(out.forecast, np.full((4,), 2.0))

  @mock.patch.object(
    timesfm3_forecaster.TimesFM3Forecaster, "_init_model", autospec=True
  )
  def test_multivariate_predict_with_covariates(self, _):
    forecaster = timesfm3_forecaster.TimesFM3Forecaster(self.config)
    forecaster.model = FakeModel()
    forecaster.device = torch.device("cpu")

    ctx = np.array(
      [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
      ]
    )
    po_cov = np.array([1.0, 1.0, 1.0, 1.0])
    pf_cov = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    out = forecaster.predict(
      ctx,
      horizon=4,
      past_only_covariates=po_cov,
      past_future_covariates=pf_cov,
      return_quantiles=True,
    )
    self.assertEqual(out.forecast.shape, (2, 4))
    self.assertEqual(out.quantiles.shape, (2, 4, 3))

  def test_query_format_padding_and_truncation(self):
    # Test padding when context_length < context_len
    q_short = timesfm3_forecaster._Query(
      horizon=4,
      targets=np.array([[1.0, 2.0]]),
      past_only_covariates=np.array([[5.0, 6.0]]),
      past_future_covariates=np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )
    hor, tgt, mask, po, pf = q_short.format(context_len=4)
    self.assertEqual(hor, 4)
    self.assertEqual(tgt.shape, (1, 4))
    np.testing.assert_array_equal(tgt, np.array([[0.0, 0.0, 1.0, 2.0]]))
    np.testing.assert_array_equal(mask, np.array([True, True, False, False]))
    np.testing.assert_array_equal(po, np.array([[0.0, 0.0, 5.0, 6.0]]))
    self.assertEqual(pf.shape, (1, 8))  # 4 ctx + 4 horizon

    # Test truncation when context_length > context_len
    q_long = timesfm3_forecaster._Query(
      horizon=2,
      targets=np.array([[1.0, 2.0, 3.0, 4.0]]),
      past_only_covariates=np.array([[10.0, 20.0, 30.0, 40.0]]),
      past_future_covariates=np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )
    hor, tgt, mask, po, pf = q_long.format(context_len=2)
    self.assertEqual(hor, 2)
    self.assertEqual(tgt.shape, (1, 2))
    np.testing.assert_array_equal(tgt, np.array([[3.0, 4.0]]))
    np.testing.assert_array_equal(mask, np.array([False, False]))
    np.testing.assert_array_equal(po, np.array([[30.0, 40.0]]))

  def test_from_pretrained_local_directory(self):
    resblock_config = configs.ResidualBlockConfig(
      hidden_dims=16,
      output_dims=16,
      use_bias=False,
      activation="relu",
    )
    transformer_config = configs.StackedTransformersConfig(
      num_layers=1,
      transformer=configs.TransformerConfig(
        model_dims=16,
        hidden_dims=16,
        num_heads=2,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_rope_seq=True,
        use_rope_var=False,
        use_bias=False,
        ff_activation="relu",
        deterministic=True,
      ),
    )
    model = torch_model_lib.TimesFM3Torch(
      input_patch_len=8,
      output_patch_len=16,
      quantiles=[0.1, 0.5, 0.9],
      residual_block_config=resblock_config,
      transformer_config=transformer_config,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      model.save_pretrained(tmpdir)
      forecaster = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(
        tmpdir, device="cpu"
      )
      self.assertEqual(forecaster.config.input_patch_length, 8)
      self.assertEqual(forecaster.config.output_patch_length, 16)
      ctx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      out = forecaster.predict(ctx, horizon=8)
      self.assertEqual(out.forecast.shape, (8,))

  def test_local_checkpoint_file_uses_published_architecture(self):
    # A local .safetensors/.pth file has no config.json, so _make_torch_model
    # supplies the architecture. It must match google/timesfm-3.0-pytorch's
    # config.json (use_rope_var=False). RoPE has no weights, so a mismatch
    # loads without error and silently changes multivariate and covariate
    # forecasts.
    with mock.patch.object(torch_model_lib, "TimesFM3Torch") as model_cls:
      timesfm3_forecaster._make_torch_model(timesfm3_forecaster._ModelConfig())
    transformer = model_cls.call_args.kwargs["transformer_config"].transformer
    self.assertTrue(transformer.use_rope_seq)
    self.assertFalse(transformer.use_rope_var)


class _TinyRealModelMixin:
  """Builds a small randomly initialized TimesFM3Torch for loading tests."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.resblock_config = configs.ResidualBlockConfig(
      hidden_dims=16, output_dims=16, use_bias=False, activation="relu"
    )
    self.transformer_config = configs.StackedTransformersConfig(
      num_layers=2,
      transformer=configs.TransformerConfig(
        model_dims=16,
        hidden_dims=16,
        num_heads=2,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_rope_seq=True,
        use_rope_var=False,
        use_bias=False,
        ff_activation="relu",
        deterministic=True,
      ),
    )
    self.model = torch_model_lib.TimesFM3Torch(
      input_patch_len=8,
      output_patch_len=16,
      quantiles=[0.1, 0.5, 0.9],
      residual_block_config=self.resblock_config,
      transformer_config=self.transformer_config,
    )
    self.model.pre_transformer_resblock.set_input_dims(2 * (8 + 16))
    for p in self.model.parameters():
      torch.nn.init.normal_(p, std=0.2)
    self.context = np.sin(np.arange(64, dtype=np.float64) / 3.0) * 5.0 + 10.0


class TimesFM3ForecasterUseSdpaTest(_TinyRealModelMixin, unittest.TestCase):
  """ModelConfig.use_sdpa must reach every attention module."""

  def _assert_use_sdpa(self, forecaster, expected):
    layers = forecaster.model.transformer_stack.layers
    self.assertEqual(len(layers), 2)
    for layer in layers:
      self.assertIs(layer.seq_attn.use_sdpa, expected)
      self.assertIs(layer.var_attn.use_sdpa, expected)

  def test_use_sdpa_local_checkpoint_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = f"{tmpdir}/model.pt"
      torch.save(self.model.state_dict(), path)
      outputs = {}
      for use_sdpa in (True, False):
        forecaster = timesfm3_forecaster.TimesFM3Forecaster(
          checkpoint_path=path,
          device="cpu",
          input_patch_length=8,
          output_patch_length=16,
          quantiles=[0.1, 0.5, 0.9],
          median_quantile_index=1,
          residual_block_config=self.resblock_config,
          transformer_config=self.transformer_config,
          use_sdpa=use_sdpa,
        )
        self._assert_use_sdpa(forecaster, use_sdpa)
        outputs[use_sdpa] = forecaster.predict(self.context, horizon=16).forecast
    np.testing.assert_allclose(outputs[True], outputs[False], rtol=1e-4, atol=1e-4)

  def test_use_sdpa_from_pretrained(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self.model.save_pretrained(tmpdir)
      outputs = {}
      for use_sdpa in (True, False):
        forecaster = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(
          tmpdir, device="cpu", use_sdpa=use_sdpa
        )
        self._assert_use_sdpa(forecaster, use_sdpa)
        outputs[use_sdpa] = forecaster.predict(self.context, horizon=16).forecast
    np.testing.assert_allclose(outputs[True], outputs[False], rtol=1e-4, atol=1e-4)


class TimesFM3ForecasterPretrainedOverridesTest(_TinyRealModelMixin, unittest.TestCase):
  """from_pretrained kwargs must be honoured or rejected, never ignored (#502)."""

  def _load(self, model, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
      model.save_pretrained(tmpdir)
      return timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(
        tmpdir, device="cpu", **kwargs
      )

  def _forecast(self, forecaster):
    return forecaster.predict(self.context, horizon=16).forecast

  def test_disabling_variate_attention_changes_forecast(self):
    default = self._load(self.model)
    ablated = self._load(self.model, use_variate_attention=False)

    self.assertFalse(ablated.config.use_variate_attention)
    self.assertFalse(ablated.model.use_variate_attention)
    for layer in ablated.model.transformer_stack.layers:
      self.assertFalse(layer.use_variate_attention)
    self.assertFalse(np.allclose(self._forecast(default), self._forecast(ablated)))

  def test_disabling_variate_attention_matches_model_built_without_it(self):
    ablated = self._load(self.model, use_variate_attention=False)

    reference = torch_model_lib.TimesFM3Torch(
      input_patch_len=8,
      output_patch_len=16,
      quantiles=[0.1, 0.5, 0.9],
      residual_block_config=self.resblock_config,
      transformer_config=self.transformer_config,
      use_variate_attention=False,
    )
    reference.pre_transformer_resblock.set_input_dims(2 * (8 + 16))
    state = {k: v for k, v in self.model.state_dict().items() if "var_attn" not in k}
    reference.load_state_dict(state)

    np.testing.assert_allclose(
      self._forecast(ablated),
      self._forecast(self._load(reference)),
      rtol=1e-5,
      atol=1e-5,
    )

  def test_runtime_option_overrides_checkpoint(self):
    forecaster = self._load(self.model, use_stitching=False, value_clip=50.0)
    self.assertFalse(forecaster.model.use_stitching)
    self.assertFalse(forecaster.config.use_stitching)
    self.assertEqual(forecaster.model.value_clip, 50.0)

  def test_unpassed_options_follow_checkpoint(self):
    # A ModelConfig default is not a request: the checkpoint's value wins.
    self.model.set_runtime_options(use_stitching=False)
    forecaster = self._load(self.model)
    self.assertFalse(forecaster.model.use_stitching)
    self.assertFalse(forecaster.config.use_stitching)

  def test_enabling_variate_attention_without_weights_raises(self):
    model = torch_model_lib.TimesFM3Torch(
      input_patch_len=8,
      output_patch_len=16,
      quantiles=[0.1, 0.5, 0.9],
      residual_block_config=self.resblock_config,
      transformer_config=self.transformer_config,
      use_variate_attention=False,
    )
    with self.assertRaisesRegex(ValueError, "use_variate_attention"):
      self._load(model, use_variate_attention=True)

  def test_mismatched_architecture_option_raises(self):
    with self.assertRaisesRegex(ValueError, "quantiles"):
      self._load(self.model, quantiles=[0.1, 0.9])
    with self.assertRaisesRegex(ValueError, "output_patch_length"):
      self._load(self.model, output_patch_length=32)

  def test_matching_architecture_option_is_accepted(self):
    forecaster = self._load(
      self.model,
      quantiles=[0.1, 0.5, 0.9],
      transformer_config=self.transformer_config,
      use_sdpa=False,
    )
    self.assertEqual(forecaster.config.quantiles, [0.1, 0.5, 0.9])


class _RecordingFakeModel(FakeModel):
  """FakeModel that records decode inputs and infers horizon like the real model."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.calls = []

  def decode(
    self,
    target,
    autoregressive_index=0,
    horizon=8,
    past_only_covariates=None,
    past_future_covariates=None,
    past_only_mask=None,
    past_future_mask=None,
    mask=None,
  ):
    del past_only_mask, past_future_mask
    if past_future_covariates is not None:
      horizon = past_future_covariates.shape[-1] - target.shape[-1]
    self.calls.append(
      {
        "target": target.clone(),
        "past_future_covariates": (
          None
          if past_future_covariates is None
          else past_future_covariates.clone()
        ),
        "past_only_covariates": (
          None
          if past_only_covariates is None
          else past_only_covariates.clone()
        ),
        "horizon": horizon,
      }
    )
    return super().decode(target, autoregressive_index, horizon, mask=mask)


class TimesFM3ForecasterCovariateWindowTest(unittest.TestCase):
  """Regression tests for past-future covariate windows under truncation."""

  def setUp(self):
    super().setUp()
    self.config = timesfm3_forecaster._ModelConfig(
      checkpoint_path="/fake/path",
      per_core_batch_size=2,
      input_patch_length=8,
      output_patch_length=8,
      median_quantile_index=1,
    )
    self._max_context = timesfm3_forecaster._MAX_CONTEXT_LENGTH
    timesfm3_forecaster._MAX_CONTEXT_LENGTH = 16

  def tearDown(self):
    timesfm3_forecaster._MAX_CONTEXT_LENGTH = self._max_context
    super().tearDown()

  def _forecaster(self):
    with mock.patch.object(
      timesfm3_forecaster.TimesFM3Forecaster, "_init_model", autospec=True
    ):
      forecaster = timesfm3_forecaster.TimesFM3Forecaster(self.config)
    forecaster.model = _RecordingFakeModel()
    forecaster.device = torch.device("cpu")
    return forecaster

  def test_truncated_context_keeps_covariate_window_aligned(self):
    # horizon=5 is not a multiple of output_patch_length=8, and the context
    # (40) exceeds the model context (16), so the query is truncated.
    forecaster = self._forecaster()
    context = np.arange(40, dtype=np.float32)
    pf_cov = np.arange(45, dtype=np.float32)  # context + horizon
    outputs = list(
      forecaster.predict_batch(
        contexts=[context], horizon=5, past_future_covariates=[pf_cov]
      )
    )
    call = forecaster.model.calls[-1]
    target = call["target"][0, 0]
    pf = call["past_future_covariates"][0, 0]
    # The covariate window must start at the same time step as the target
    # window and its future part must be exactly the requested horizon.
    self.assertEqual(float(target[0]), float(pf[0]))
    self.assertEqual(pf.shape[-1], target.shape[-1] + 5)
    self.assertEqual(call["horizon"], 5)
    self.assertEqual(outputs[0].forecast.shape, (5,))

  def test_mixed_length_batch_with_covariates(self):
    forecaster = self._forecaster()
    long_ctx = np.arange(40, dtype=np.float32)
    short_ctx = np.arange(100, 112, dtype=np.float32)
    outputs = list(
      forecaster.predict_batch(
        contexts=[long_ctx, short_ctx],
        horizon=5,
        past_future_covariates=[
          np.arange(45, dtype=np.float32),
          np.arange(100, 117, dtype=np.float32),
        ],
      )
    )
    self.assertEqual([o.forecast.shape for o in outputs], [(5,), (5,)])

  def test_mixed_none_and_pf_covariates(self):
    # A query without past-future covariates batched with one that has
    # them: the placeholder must have the decode-expected width (matching
    # the real PF arrays in the batch), or np.stack fails on lengths.
    forecaster = self._forecaster()
    long_ctx = np.arange(40, dtype=np.float32)
    short_ctx = np.arange(100, 112, dtype=np.float32)
    outputs = list(
      forecaster.predict_batch(
        contexts=[long_ctx, short_ctx],
        horizon=5,
        past_future_covariates=[np.arange(45, dtype=np.float32), None],
      )
    )
    self.assertEqual([o.forecast.shape for o in outputs], [(5,), (5,)])
    call = forecaster.model.calls[-1]
    pf = call["past_future_covariates"]
    # decode must see a uniform PF width across the batch. The batch
    # context is capped by _MAX_CONTEXT_LENGTH(16, set in setUp), so the
    # real PF is context(16) + requested horizon(5) = 21; the placeholder
    # matches that width.
    self.assertEqual(pf.shape[0], 2)
    self.assertEqual(pf.shape[-1], 21)
    pf_real = pf[0, 0]
    pf_placeholder = pf[1, 0]
    self.assertEqual(pf_real.shape[-1], pf_placeholder.shape[-1])
    # The placeholder's future part (the requested horizon, 5) is zeros;
    # decode infers horizon 5 from the width.
    np.testing.assert_array_equal(pf_placeholder[-5:], np.zeros((5,), dtype=np.float32))
    self.assertEqual(call["horizon"], 5)

  def test_mixed_none_and_multichannel_covariates(self):
    # Covariate channel count need not equal the number of target variates:
    # a 2-channel past-future covariate and a 2-channel past-only covariate
    # batched with a query that has neither must still stack.
    forecaster = self._forecaster()
    ctx_a = np.arange(12, dtype=np.float32)
    ctx_b = np.arange(100, 112, dtype=np.float32)
    pf_a = np.stack([np.arange(17, dtype=np.float32)] * 2)
    po_a = np.stack([np.arange(12, dtype=np.float32)] * 2)
    outputs = list(
      forecaster.predict_batch(
        contexts=[ctx_a, ctx_b],
        horizon=5,
        past_only_covariates=[po_a, None],
        past_future_covariates=[pf_a, None],
      )
    )
    self.assertEqual([o.forecast.shape for o in outputs], [(5,), (5,)])
    call = forecaster.model.calls[-1]
    # batch context 16 (12 rounded up to the patch), 2 channels, future 5.
    self.assertEqual(tuple(call["past_future_covariates"].shape), (2, 2, 21))
    self.assertEqual(tuple(call["past_only_covariates"].shape), (2, 2, 16))
    np.testing.assert_array_equal(
      call["past_future_covariates"][1].numpy(), np.zeros((2, 21), np.float32)
    )
    np.testing.assert_array_equal(
      call["past_only_covariates"][1].numpy(), np.zeros((2, 16), np.float32)
    )

  def test_mixed_none_covariates_multivariate_target(self):
    # Three target variates with a 2-channel past-future covariate on one
    # query and none on the other.
    forecaster = self._forecaster()
    ctx_a = np.stack([np.arange(12, dtype=np.float32)] * 3)
    ctx_b = np.stack([np.arange(100, 112, dtype=np.float32)] * 3)
    pf_a = np.stack([np.arange(17, dtype=np.float32)] * 2)
    outputs = list(
      forecaster.predict_batch(
        contexts=[ctx_a, ctx_b], horizon=5, past_future_covariates=[pf_a, None]
      )
    )
    self.assertEqual([o.forecast.shape for o in outputs], [(3, 5), (3, 5)])
    call = forecaster.model.calls[-1]
    self.assertEqual(tuple(call["target"].shape), (2, 3, 16))
    self.assertEqual(tuple(call["past_future_covariates"].shape), (2, 2, 21))

  def test_predict_batch_accepts_2d_array(self):
    forecaster = self._forecaster()
    outputs = list(
      forecaster.predict_batch(
        contexts=np.random.rand(3, 12).astype(np.float32), horizon=4
      )
    )
    self.assertEqual(len(outputs), 3)

  def test_evaluator_predict_batch_accepts_2d_array(self):
    # Same as above for TimesFM3Evaluator, which had its own `if not contexts`
    # guard. Cover both the default and the univariate=True paths.
    with mock.patch.object(
      evaluator.TimesFM3Evaluator, "_init_model", autospec=True
    ):
      ev = evaluator.TimesFM3Evaluator(self.config)
    ev.model = _RecordingFakeModel()
    ev.device = torch.device("cpu")
    contexts = np.random.rand(3, 12).astype(np.float32)
    for univariate in (False, True):
      outputs = list(
        ev.predict_batch(contexts=contexts, horizon=4, univariate=univariate)
      )
      self.assertEqual([o.forecast.shape for o in outputs], [(4,)] * 3)


class TimesFM3ForecasterCovariatePlaceholderTest(unittest.TestCase):
  """A query without covariates must not depend on its batch mates' covariates."""

  def _forecaster(self):
    torch.manual_seed(0)
    model = torch_model_lib.TimesFM3Torch(
      input_patch_len=8,
      output_patch_len=16,
      quantiles=[0.1, 0.5, 0.9],
      residual_block_config=configs.ResidualBlockConfig(
        hidden_dims=16, output_dims=16, use_bias=False, activation="relu"
      ),
      transformer_config=configs.StackedTransformersConfig(
        num_layers=1,
        transformer=configs.TransformerConfig(
          model_dims=16,
          hidden_dims=16,
          num_heads=2,
          attention_norm="rms",
          feedforward_norm="rms",
          qk_norm="rms",
          use_rope_seq=True,
          use_rope_var=False,
          use_bias=False,
          ff_activation="relu",
          deterministic=True,
        ),
      ),
    )
    # Perturb every weight (norm scales are zero-initialized) so the
    # covariate channels actually influence the output.
    with torch.no_grad():
      for p in model.parameters():
        p.add_(torch.randn_like(p) * 0.3)
    with tempfile.TemporaryDirectory() as tmpdir:
      model.save_pretrained(tmpdir)
      forecaster = timesfm3_forecaster.TimesFM3Forecaster.from_pretrained(
        tmpdir, device="cpu"
      )
    forecaster.config.per_core_batch_size = 2
    return forecaster

  def test_no_covariate_query_unaffected_by_batch_mate(self):
    forecaster = self._forecaster()
    rng = np.random.default_rng(0)
    horizon = 16
    # Same context length, so no padding differs between the two runs.
    ctx_a = (np.sin(np.arange(64) / 3) + 5).astype(np.float32)
    ctx_b = (np.cos(np.arange(64) / 5) * 3).astype(np.float32)
    pf_b = rng.normal(size=64 + horizon).astype(np.float32)
    po_b = rng.normal(size=64).astype(np.float32)

    alone = next(iter(forecaster.predict_batch([ctx_a], horizon=horizon)))
    for kwargs in (
      {"past_future_covariates": [None, pf_b]},
      {"past_only_covariates": [None, po_b]},
    ):
      with self.subTest(**{k: "set" for k in kwargs}):
        batched = list(
          forecaster.predict_batch([ctx_a, ctx_b], horizon=horizon, **kwargs)
        )
        np.testing.assert_allclose(
          batched[0].forecast, alone.forecast, rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
  unittest.main()
