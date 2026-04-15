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

"""Tests for PyTorch layer building blocks: ResidualBlock, RMSNorm,
RandomFourierFeatures.

These layers are the atoms of the TimesFM architecture. Verifying their
output shapes, numerical properties, and failure modes protects against
regressions during refactors.  All tests use small dimensions and run
on CPU — no model checkpoint or GPU required.
"""

import torch
import pytest

from timesfm.configs import RandomFourierFeaturesConfig, ResidualBlockConfig
from timesfm.torch.dense import RandomFourierFeatures, ResidualBlock
from timesfm.torch.normalization import RMSNorm


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------


class TestResidualBlock:
  """Tests for the residual block: hidden → activation → output + skip."""

  @pytest.fixture
  def swish_block(self):
    """A small residual block with SiLU/Swish activation (matches TimesFM)."""
    cfg = ResidualBlockConfig(
      input_dims=16,
      hidden_dims=32,
      output_dims=8,
      use_bias=True,
      activation="swish",
    )
    return ResidualBlock(cfg)

  def test_output_shape(self, swish_block):
    """Output must have the config's ``output_dims`` as the last dimension,
    regardless of input batch shape."""
    x = torch.randn(4, 16)
    out = swish_block(x)
    assert out.shape == (4, 8)

  def test_output_shape_3d(self, swish_block):
    """The block must handle (batch, seq, features) inputs — the layout
    used when processing patched time series."""
    x = torch.randn(2, 10, 16)
    out = swish_block(x)
    assert out.shape == (2, 10, 8)

  def test_residual_connection_nonzero(self):
    """The residual connection must contribute to the output.

    We verify this by comparing the output when the hidden path is
    zeroed out vs. the full output.
    """
    cfg = ResidualBlockConfig(
      input_dims=8,
      hidden_dims=16,
      output_dims=8,
      use_bias=False,
      activation="none",
    )
    block = ResidualBlock(cfg)
    x = torch.randn(2, 8)

    with torch.no_grad():
      # Residual path only: zero out hidden and output layers.
      block.hidden_layer.weight.zero_()
      block.output_layer.weight.zero_()
      residual_only = block(x)

      # Must equal the residual layer output.
      expected = block.residual_layer(x)
      torch.testing.assert_close(residual_only, expected)

  @pytest.mark.parametrize("activation", ["relu", "swish", "none"])
  def test_all_activations_produce_valid_output(self, activation):
    """All supported activations must produce finite, non-NaN output."""
    cfg = ResidualBlockConfig(
      input_dims=8,
      hidden_dims=16,
      output_dims=8,
      use_bias=True,
      activation=activation,
    )
    block = ResidualBlock(cfg)
    x = torch.randn(4, 8)
    out = block(x)
    assert not torch.any(torch.isnan(out))
    assert not torch.any(torch.isinf(out))

  def test_invalid_activation_raises(self):
    """Unsupported activation must raise ``ValueError`` immediately —
    fail fast rather than producing garbage at inference time."""
    cfg = ResidualBlockConfig(
      input_dims=8,
      hidden_dims=16,
      output_dims=8,
      use_bias=True,
      activation="gelu",
    )
    with pytest.raises(ValueError, match="not supported"):
      ResidualBlock(cfg)

  def test_gradient_flows_through_both_paths(self):
    """Gradients must reach both the main path and the residual path.

    Dead gradients on either path would prevent the layer from learning.
    """
    cfg = ResidualBlockConfig(
      input_dims=8,
      hidden_dims=16,
      output_dims=8,
      use_bias=True,
      activation="swish",
    )
    block = ResidualBlock(cfg)
    x = torch.randn(2, 8, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()

    assert block.hidden_layer.weight.grad is not None
    assert block.residual_layer.weight.grad is not None
    assert torch.any(block.hidden_layer.weight.grad != 0)
    assert torch.any(block.residual_layer.weight.grad != 0)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
  """Tests for RMS normalization used in transformer attention/FF blocks."""

  def test_output_shape_preserved(self):
    """RMSNorm must not change the tensor shape."""
    norm = RMSNorm(num_features=64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape

  def test_zero_scale_produces_zeros(self):
    """With default scale (initialized to zeros), output must be all zeros.

    This is a critical initialization property: at init, each transformer
    layer's post-norm effectively passes through zeros, relying on the
    residual connection to carry signal.
    """
    norm = RMSNorm(num_features=8)
    # scale is initialized to zeros by default.
    x = torch.randn(4, 8)
    out = norm(x)
    torch.testing.assert_close(out, torch.zeros_like(out))

  def test_unit_scale_preserves_rms_magnitude(self):
    """With scale = 1, output should have approximately unit RMS along
    the feature dimension — that's the point of RMS normalization."""
    norm = RMSNorm(num_features=64)
    with torch.no_grad():
      norm.scale.fill_(1.0)

    x = torch.randn(8, 64) * 100  # large magnitude
    out = norm(x)

    rms = torch.sqrt(torch.mean(out**2, dim=-1))
    # After normalization, RMS should be close to 1.0.
    torch.testing.assert_close(
      rms,
      torch.ones(8),
      atol=0.1,
      rtol=0.1,
    )

  def test_no_nan_on_zero_input(self):
    """A zero-valued input must not cause NaN (epsilon prevents div-by-0)."""
    norm = RMSNorm(num_features=8, epsilon=1e-6)
    with torch.no_grad():
      norm.scale.fill_(1.0)
    x = torch.zeros(2, 8)
    out = norm(x)
    assert not torch.any(torch.isnan(out))


# ---------------------------------------------------------------------------
# RandomFourierFeatures
# ---------------------------------------------------------------------------


class TestRandomFourierFeatures:
  """Tests for the random Fourier feature layer."""

  def test_output_shape(self):
    """Output dims must be exactly ``config.output_dims``."""
    cfg = RandomFourierFeaturesConfig(
      input_dims=8,
      output_dims=32,
      projection_stddev=1.0,
      use_bias=True,
    )
    layer = RandomFourierFeatures(cfg)
    x = torch.randn(4, 8)
    out = layer(x)
    assert out.shape == (4, 32)

  def test_output_dims_not_multiple_of_4_raises(self):
    """The four Fourier components (cos, sin, sq_wave_1, sq_wave_2)
    require ``output_dims`` to be divisible by 4."""
    cfg = RandomFourierFeaturesConfig(
      input_dims=8,
      output_dims=30,  # not divisible by 4
      projection_stddev=1.0,
      use_bias=True,
    )
    with pytest.raises(ValueError, match="multiple of 4"):
      RandomFourierFeatures(cfg)

  def test_fourier_components_bounded(self):
    """cos and sin outputs are bounded in [-1, 1]; sign outputs are
    bounded in {-1, 0, 1}. The total Fourier part (before residual)
    is thus bounded. We verify the output stays finite."""
    cfg = RandomFourierFeaturesConfig(
      input_dims=8,
      output_dims=32,
      projection_stddev=1.0,
      use_bias=False,
    )
    layer = RandomFourierFeatures(cfg)
    x = torch.randn(16, 8) * 10  # moderately large input
    out = layer(x)
    assert not torch.any(torch.isnan(out))
    assert not torch.any(torch.isinf(out))

  def test_3d_input_supported(self):
    """The layer must handle (batch, seq, features) tensors."""
    cfg = RandomFourierFeaturesConfig(
      input_dims=8,
      output_dims=16,
      projection_stddev=1.0,
      use_bias=True,
    )
    layer = RandomFourierFeatures(cfg)
    x = torch.randn(2, 5, 8)
    out = layer(x)
    assert out.shape == (2, 5, 16)
