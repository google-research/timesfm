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

"""Tests for TimesFM configuration dataclasses.

These tests verify that config dataclasses enforce immutability, compose
correctly, and carry the exact default values the model implementation
relies on. Catching a silent default-value drift here prevents subtle
inference regressions that would otherwise only surface as degraded
forecast quality.
"""

import dataclasses

import pytest

from timesfm.configs import (
  ForecastConfig,
  RandomFourierFeaturesConfig,
  ResidualBlockConfig,
  StackedTransformersConfig,
  TransformerConfig,
)


# ---------------------------------------------------------------------------
# ForecastConfig
# ---------------------------------------------------------------------------


class TestForecastConfig:
  """Tests for ForecastConfig — the primary user-facing configuration."""

  def test_defaults_match_safe_inference_settings(self):
    """Default config must be conservative: no normalization, no fancy heads.

    These defaults are what users get when they call ``ForecastConfig()``
    without arguments. Changing them silently would break all existing
    code that relies on the defaults.
    """
    cfg = ForecastConfig()
    assert cfg.max_context == 0
    assert cfg.max_horizon == 0
    assert cfg.normalize_inputs is False
    assert cfg.per_core_batch_size == 1
    assert cfg.use_continuous_quantile_head is False
    assert cfg.force_flip_invariance is True
    assert cfg.infer_is_positive is True
    assert cfg.fix_quantile_crossing is False
    assert cfg.return_backcast is False

  def test_frozen_prevents_mutation(self):
    """Configs are frozen dataclasses — mutating them must raise.

    This is critical because ``compile()`` captures the config object and
    the compiled decode closure relies on its values never changing.
    """
    cfg = ForecastConfig(max_context=512)
    with pytest.raises(dataclasses.FrozenInstanceError):
      cfg.max_context = 1024

  def test_replace_creates_independent_copy(self):
    """``dataclasses.replace`` must yield a new object with updated fields.

    The compile path uses ``replace`` to adjust context/horizon to valid
    multiples; the original config must remain untouched.
    """
    original = ForecastConfig(max_context=512, max_horizon=128)
    replaced = dataclasses.replace(original, max_context=1024)

    assert replaced.max_context == 1024
    assert replaced.max_horizon == 128  # untouched
    assert original.max_context == 512  # original unchanged

  def test_equality_is_structural(self):
    """Two configs with identical fields must be equal (value semantics)."""
    a = ForecastConfig(max_context=256, normalize_inputs=True)
    b = ForecastConfig(max_context=256, normalize_inputs=True)
    assert a == b

  def test_inequality_on_any_field_difference(self):
    """A single differing field must break equality."""
    a = ForecastConfig(max_context=256)
    b = ForecastConfig(max_context=512)
    assert a != b


# ---------------------------------------------------------------------------
# ResidualBlockConfig
# ---------------------------------------------------------------------------


class TestResidualBlockConfig:
  """Tests for ResidualBlockConfig used by tokenizer and output projections."""

  def test_frozen_prevents_mutation(self):
    cfg = ResidualBlockConfig(
      input_dims=64,
      hidden_dims=128,
      output_dims=128,
      use_bias=True,
      activation="swish",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
      cfg.input_dims = 32

  def test_activation_accepts_all_valid_literals(self):
    """All three activation modes must be constructable without error."""
    for act in ("relu", "swish", "none"):
      cfg = ResidualBlockConfig(
        input_dims=8,
        hidden_dims=16,
        output_dims=8,
        use_bias=False,
        activation=act,
      )
      assert cfg.activation == act


# ---------------------------------------------------------------------------
# TransformerConfig & StackedTransformersConfig
# ---------------------------------------------------------------------------


class TestTransformerConfig:
  """Tests for TransformerConfig — architecture-level hyperparameters."""

  def test_model_dims_must_be_divisible_by_num_heads(self):
    """The model instantiation will fail if this invariant is broken.

    We verify the config at least *carries* the right values that the
    TimesFM 2.5 definition uses (1280 dims, 16 heads → 80 head_dim).
    """
    cfg = TransformerConfig(
      model_dims=1280,
      hidden_dims=1280,
      num_heads=16,
      attention_norm="rms",
      feedforward_norm="rms",
      qk_norm="rms",
      use_bias=False,
      use_rotary_position_embeddings=True,
      ff_activation="swish",
      fuse_qkv=True,
    )
    assert cfg.model_dims % cfg.num_heads == 0
    assert cfg.model_dims // cfg.num_heads == 80  # head_dim

  def test_stacked_config_composes_correctly(self):
    """StackedTransformersConfig must wrap a TransformerConfig cleanly."""
    xf = TransformerConfig(
      model_dims=64,
      hidden_dims=64,
      num_heads=4,
      attention_norm="rms",
      feedforward_norm="rms",
      qk_norm="none",
      use_bias=True,
      use_rotary_position_embeddings=False,
      ff_activation="relu",
      fuse_qkv=False,
    )
    stacked = StackedTransformersConfig(num_layers=6, transformer=xf)
    assert stacked.num_layers == 6
    assert stacked.transformer is xf
    assert stacked.transformer.model_dims == 64


# ---------------------------------------------------------------------------
# RandomFourierFeaturesConfig
# ---------------------------------------------------------------------------


class TestRandomFourierFeaturesConfig:
  """Tests for RandomFourierFeaturesConfig."""

  def test_frozen_prevents_mutation(self):
    cfg = RandomFourierFeaturesConfig(
      input_dims=32,
      output_dims=64,
      projection_stddev=1.0,
      use_bias=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
      cfg.output_dims = 128
