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

import os
import tempfile

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.timesfm_2p5.timesfm_2p5_flax import TimesFM_2p5_200M_flax


class TestModelLoading:
  """Tests to verify model instantiation, loading, and compatibility."""

  def test_torch_load_checkpoint_and_from_pretrained_local(self):
    """Verifies that PyTorch load_checkpoint and from_pretrained work locally."""
    # 1. Instantiate the model wrapper with compilation disabled
    tfm = TimesFM_2p5_200M_torch(torch_compile=False)

    with tempfile.TemporaryDirectory() as tmpdir:
      # 2. Save the model's randomly-initialized weights
      tfm._save_pretrained(tmpdir)

      # Verify weights file is written
      weights_path = os.path.join(tmpdir, "model.safetensors")
      assert os.path.exists(weights_path)

      # 3. Verify that load_checkpoint works from the temp directory path
      tfm2 = TimesFM_2p5_200M_torch(torch_compile=False)
      tfm2.load_checkpoint(tmpdir, torch_compile=False)

      # 4. Verify that from_pretrained works with a local directory path
      # and accepts/ignores extra kwargs (like proxies) without raising TypeError
      tfm3 = TimesFM_2p5_200M_torch.from_pretrained(
          tmpdir,
          torch_compile=False,
          proxies={"http": "http://dummy.proxy"},
          custom_kwarg="dummy_value",
      )
      assert tfm3 is not None
      assert not tfm3.torch_compile

      # 5. Run a simple prediction step to verify the loaded model performs forward pass
      import numpy as np
      inputs = [np.random.randn(32)]
      forecasts = tfm3.model.forecast_naive(horizon=10, inputs=inputs)
      assert len(forecasts) == 1
      assert forecasts[0].shape == (10, 10)

  def test_flax_model_init_kwargs(self):
    """Verifies that Flax model wrapper constructor accepts arbitrary kwargs."""
    tfm = TimesFM_2p5_200M_flax(
        proxies={"http": "http://dummy.proxy"},
        custom_kwarg="dummy_value",
    )
    assert tfm is not None
