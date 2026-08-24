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


if __name__ == "__main__":
  unittest.main()
