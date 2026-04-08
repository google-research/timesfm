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

import torch
from torch import nn

from timesfm.timesfm_2p5 import timesfm_2p5_torch


class _CompiledModel:

  def __init__(self):
    self.eval_called = False

  def eval(self):
    self.eval_called = True
    return self


class _DummyModel(nn.Module):
  load_checkpoint = timesfm_2p5_torch.TimesFM_2p5_200M_torch_module.load_checkpoint

  def __init__(self):
    super().__init__()
    self.device = torch.device("cpu")
    self.state_dict_calls = []
    self.to_calls = []
    self.eval_called = False

  def load_state_dict(self, tensors, strict=True):
    self.state_dict_calls.append((tensors, strict))

  def to(self, device):
    self.to_calls.append(device)
    return self

  def eval(self):
    self.eval_called = True
    return self


class _DummyTimesFM(timesfm_2p5_torch.TimesFM_2p5_200M_torch):

  def __init__(self, torch_compile=True, config=None):
    self.model = _DummyModel()
    self.torch_compile = torch_compile
    if config is not None:
      self._hub_mixin_config = config


def test_from_pretrained_retains_compiled_model(monkeypatch):
  compiled_model = _CompiledModel()

  monkeypatch.setattr(timesfm_2p5_torch, "load_file", lambda path: {"w": 1})
  monkeypatch.setattr(
    timesfm_2p5_torch,
    "hf_hub_download",
    lambda **kwargs: "C:/fake/model.safetensors",
  )
  monkeypatch.setattr(torch, "compile", lambda model: compiled_model)

  instance = _DummyTimesFM._from_pretrained(
    model_id="google/timesfm-2.5-200m-pytorch",
    revision="main",
    cache_dir="C:/cache",
    force_download=False,
    local_files_only=True,
    token=None,
  )

  assert instance.model is compiled_model
  assert compiled_model.eval_called
