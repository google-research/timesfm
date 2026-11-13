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

"""Tests for PyTorch TimesFM3Torch model."""

import unittest

import torch

from . import configs
from . import model as torch_model_lib


class TimesFM3TorchTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.resblock_config = configs.ResidualBlockConfig(
      hidden_dims=32,
      output_dims=32,
      use_bias=False,
      activation="relu",
      dropout=0.0,
    )
    self.transformer_config = configs.StackedTransformersConfig(
      num_layers=2,
      use_remat=False,
      transformer=configs.TransformerConfig(
        model_dims=32,
        hidden_dims=32,
        num_heads=4,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_rope_seq=True,
        use_rope_var=True,
        use_bias=False,
        ff_activation="relu",
        deterministic=True,
        use_sdpa=True,
      ),
    )
    self.model = torch_model_lib.TimesFM3Torch(
      input_patch_len=8,
      output_patch_len=16,
      quantiles=[0.1, 0.5, 0.9],
      residual_block_config=self.resblock_config,
      transformer_config=self.transformer_config,
      use_stitching=True,
      use_linear_detrending=True,
      use_iterative_cpm_revin=True,
      use_frozen_running_stats=False,
    )
    self.model.eval()

  def test_forward_pass_training_dict(self):
    b, v, n, p = 2, 3, 4, 8
    inputs = {
      "values": torch.randn(b, v, n, p),
      "masks": torch.zeros(b, v, n, p, dtype=torch.bool),
      "patch_segment_ids": torch.zeros(b, n, dtype=torch.long),
      "patch_positions": torch.arange(n).unsqueeze(0).repeat(b, 1),
      "patch_is_target": torch.ones(b, v, n, dtype=torch.bool),
      "patch_is_past_only": torch.zeros(b, v, n, dtype=torch.bool),
      "patch_is_past_future_covariate": torch.zeros(b, v, n, dtype=torch.bool),
    }
    with torch.no_grad():
      out = self.model(inputs)
    self.assertIn("logits", out)
    # Shape: (b, v, n, output_patch_len, num_quantiles) = (2, 3, 4, 16, 3)
    self.assertEqual(out["logits"].shape, (b, v, n, 16, 3))

  def test_decode_univariate(self):
    b, v, context_len = 2, 1, 32
    target = torch.randn(b, v, context_len)
    horizon = 32
    with torch.no_grad():
      out = self.model.decode(target=target, horizon=horizon)
    # Output shape: (b, v, horizon, num_quantiles) = (2, 1, 32, 3)
    self.assertEqual(out.shape, (b, v, horizon, 3))
    self.assertTrue(torch.isfinite(out).all().item())

  def test_decode_multivariate_with_covariates(self):
    b, v, context_len = 2, 3, 32
    target = torch.randn(b, v, context_len)
    po_cov = torch.randn(b, v, context_len)
    horizon = 32
    pf_cov = torch.randn(b, v, context_len + horizon)
    mask = torch.zeros(b, context_len, dtype=torch.bool)

    with torch.no_grad():
      out = self.model.decode(
        target=target,
        horizon=horizon,
        past_only_covariates=po_cov,
        past_future_covariates=pf_cov,
        mask=mask,
      )
    # Total variates: 3 targets + 3 past_only + 3 past_future = 9
    self.assertEqual(out.shape, (b, 9, horizon, 3))
    self.assertTrue(torch.isfinite(out).all().item())

  def test_save_and_from_pretrained(self):
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
      self.model.save_pretrained(tmpdir)
      config_path = os.path.join(tmpdir, "config.json")
      self.assertTrue(os.path.exists(config_path))
      with open(config_path) as f:
        config_dict = json.load(f)
      self.assertEqual(config_dict["input_patch_len"], 8)
      self.assertEqual(config_dict["output_patch_len"], 16)
      self.assertIn("transformer_config", config_dict)

      loaded_model = torch_model_lib.TimesFM3Torch.from_pretrained(tmpdir)
      loaded_model.eval()

      target = torch.randn(2, 1, 32)
      with torch.no_grad():
        orig_out = self.model.decode(target=target, horizon=16)
        loaded_out = loaded_model.decode(target=target, horizon=16)
      self.assertTrue(torch.allclose(orig_out, loaded_out, atol=1e-5))


def _build_model(use_sdpa: bool, v_norm: str) -> torch_model_lib.TimesFM3Torch:
  torch.manual_seed(0)
  model = torch_model_lib.TimesFM3Torch(
    input_patch_len=8,
    output_patch_len=16,
    quantiles=[0.1, 0.5, 0.9],
    residual_block_config=configs.ResidualBlockConfig(
      hidden_dims=32, output_dims=32, use_bias=False, activation="relu", dropout=0.0
    ),
    transformer_config=configs.StackedTransformersConfig(
      num_layers=2,
      use_remat=False,
      transformer=configs.TransformerConfig(
        model_dims=32,
        hidden_dims=32,
        num_heads=4,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        v_norm=v_norm,
        use_rope_seq=True,
        use_rope_var=True,
        use_bias=False,
        ff_activation="relu",
        deterministic=True,
        use_sdpa=use_sdpa,
      ),
    ),
    use_stitching=True,
    use_linear_detrending=True,
    use_iterative_cpm_revin=True,
    use_frozen_running_stats=False,
  )
  model.eval()
  return model


def _set_fast_path(model: torch_model_lib.TimesFM3Torch, enabled: bool) -> None:
  for layer in model.modules():
    if hasattr(layer, "_single_variate_fast_path"):
      layer._single_variate_fast_path = enabled


class SingleVariateFastPathTest(unittest.TestCase):
  """The V == 1 variate-attention shortcut is used and changes no forecast."""

  def test_fast_path_used_only_for_one_variate(self):
    model = _build_model(use_sdpa=True, v_norm="none")
    calls = []
    for layer in model.modules():
      if hasattr(layer, "var_attn"):
        layer.var_attn.query_proj.register_forward_hook(lambda *_: calls.append(1))

    with torch.no_grad():
      model.decode(target=torch.randn(2, 1, 32), horizon=32)
    self.assertEqual(len(calls), 0, "V == 1 should skip the variate Q projection")

    with torch.no_grad():
      model.decode(target=torch.randn(2, 2, 32), horizon=32)
    self.assertGreater(len(calls), 0, "V > 1 must still run full variate attention")

  def test_decode_matches_full_attention(self):
    b, context_len, horizon = 3, 37, 48
    torch.manual_seed(1)
    target = torch.randn(b, 1, context_len)
    target[1, 0, 20:23] = float("nan")  # gap inside the context
    # Series 0 and 2 have front padding, so whole patches are masked.
    mask = torch.zeros(b, context_len, dtype=torch.bool)
    mask[0, :13] = True
    mask[2, :30] = True

    for use_sdpa in (True, False):
      for v_norm in ("none", "rms"):
        with self.subTest(use_sdpa=use_sdpa, v_norm=v_norm):
          model = _build_model(use_sdpa=use_sdpa, v_norm=v_norm)
          with torch.no_grad():
            fast = model.decode(target=target, horizon=horizon, mask=mask)
            _set_fast_path(model, False)
            ref = model.decode(target=target, horizon=horizon, mask=mask)
          self.assertEqual(fast.shape, (b, 1, horizon, 3))
          self.assertTrue(torch.isfinite(fast).all().item())
          torch.testing.assert_close(fast, ref, rtol=0, atol=0)


if __name__ == "__main__":
  unittest.main()
