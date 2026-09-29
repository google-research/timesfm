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

"""Tests for loading TimesFM 2.5 models."""

import numpy as np
import os
import tempfile
import types

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.timesfm_2p5.timesfm_2p5_flax import TimesFM_2p5_200M_flax


class TestModelLoading:
  """Tests to verify model instantiation, loading, and compatibility."""

  def test_torch_load_checkpoint_and_from_pretrained_local(self):
    """Verifies that PyTorch load_checkpoint and from_pretrained work locally."""
    tfm = TimesFM_2p5_200M_torch(torch_compile=False)

    with tempfile.TemporaryDirectory() as tmpdir:
      tfm._save_pretrained(tmpdir)

      weights_path = os.path.join(tmpdir, "model.safetensors")
      assert os.path.exists(weights_path)

      tfm2 = TimesFM_2p5_200M_torch(torch_compile=False)
      tfm2.load_checkpoint(tmpdir, torch_compile=False)

      tfm3 = TimesFM_2p5_200M_torch.from_pretrained(
          tmpdir,
          torch_compile=False,
          proxies={"http": "http://dummy.proxy"},
          custom_kwarg="dummy_value",
      )
      assert tfm3 is not None
      assert not tfm3.torch_compile

      inputs = [np.random.randn(32)]
      forecasts = tfm3.model.forecast_naive(horizon=10, inputs=inputs)
      assert len(forecasts) == 1
      assert forecasts[0].shape == (10, 10)

  def test_torch_compile_wraps_forward(self):
    """Verifies that torch_compile=True compiles model.forward."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tfm = TimesFM_2p5_200M_torch(torch_compile=False)
      tfm._save_pretrained(tmpdir)

      tfm_compiled = TimesFM_2p5_200M_torch(torch_compile=True)
      tfm_compiled.load_checkpoint(tmpdir)

      assert not isinstance(tfm_compiled.model.forward, types.MethodType)

  def test_torch_no_compile_leaves_forward_unchanged(self):
    """Verifies that torch_compile=False leaves model.forward as a plain method."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tfm = TimesFM_2p5_200M_torch(torch_compile=False)
      tfm._save_pretrained(tmpdir)

      tfm_no_compile = TimesFM_2p5_200M_torch(torch_compile=False)
      tfm_no_compile.load_checkpoint(tmpdir)

      assert isinstance(tfm_no_compile.model.forward, types.MethodType)

  def test_flax_model_init_kwargs(self):
    """Verifies that Flax model wrapper constructor accepts arbitrary kwargs."""
    tfm = TimesFM_2p5_200M_flax(
      proxies={"http": "http://dummy.proxy"},
      custom_kwarg="dummy_value",
    )
    assert tfm is not None


class TestForecastConfigWindowSize:
  """Integration tests for window_size in ForecastConfig."""

  def test_window_size_forecast_returns_correct_shape(self):
    """Verifies that forecast() with window_size returns correct shape."""
    from timesfm import configs

    tfm = TimesFM_2p5_200M_torch(torch_compile=False)
    tfm.compile(
      forecast_config=configs.ForecastConfig(
        max_context=128,
        max_horizon=64,
        window_size=10,
      )
    )

    inputs = [np.random.randn(256) for _ in range(3)]
    points, quantiles = tfm.forecast(horizon=32, inputs=inputs)

    assert points.shape == (3, 32)
    assert quantiles.shape == (3, 32, 10)

  def test_window_size_default_zero(self):
    """Verifies that window_size=0 does not change output shape."""
    from timesfm import configs

    tfm = TimesFM_2p5_200M_torch(torch_compile=False)
    tfm.compile(
      forecast_config=configs.ForecastConfig(
        max_context=128,
        max_horizon=64,
        window_size=0,
      )
    )

    inputs = [np.random.randn(200) for _ in range(2)]
    points, quantiles = tfm.forecast(horizon=16, inputs=inputs)

    assert points.shape == (2, 16)
    assert quantiles.shape == (2, 16, 10)

  def test_window_size_covariates_raises_error(self):
    """Verifies that forecast_with_covariates errors with window_size > 0."""
    import pytest
    from timesfm import configs

    tfm = TimesFM_2p5_200M_torch(torch_compile=False)
    tfm.compile(
      forecast_config=configs.ForecastConfig(
        max_context=128,
        max_horizon=64,
        window_size=10,
        return_backcast=True,
      )
    )

    with pytest.raises(ValueError, match="window_size"):
      tfm.forecast_with_covariates(
        inputs=[np.random.randn(128)],
        static_numerical_covariates={"a": [1, 2, 3]},
      )
