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

from timesfm.timesfm_2p5 import timesfm_2p5_torch


class _DummyModel:

  def __init__(self):
    self.calls = []

  def load_checkpoint(self, path, **kwargs):
    self.calls.append((path, kwargs))


class _DummyTimesFM(timesfm_2p5_torch.TimesFM_2p5_200M_torch):

  def __init__(self, torch_compile=True, config=None):
    self.model = _DummyModel()
    self.torch_compile = torch_compile
    if config is not None:
      self._hub_mixin_config = config


def test_from_pretrained_accepts_hf_hub_v1_kwargs(monkeypatch):
  captured = {}

  def fake_hf_hub_download(**kwargs):
    captured.update(kwargs)
    return "C:/fake/model.safetensors"

  monkeypatch.setattr(timesfm_2p5_torch, "hf_hub_download", fake_hf_hub_download)

  instance = _DummyTimesFM._from_pretrained(
    model_id="google/timesfm-2.5-200m-pytorch",
    revision="main",
    cache_dir="C:/cache",
    force_download=False,
    proxies={"https": "https://proxy.example"},
    resume_download=None,
    local_files_only=True,
    token="token",
  )

  assert captured == {
    "repo_id": "google/timesfm-2.5-200m-pytorch",
    "filename": _DummyTimesFM.WEIGHTS_FILENAME,
    "revision": "main",
    "cache_dir": "C:/cache",
    "force_download": False,
    "proxies": {"https": "https://proxy.example"},
    "resume_download": None,
    "token": "token",
    "local_files_only": True,
  }
  assert instance.model.calls == [
    ("C:/fake/model.safetensors", {"torch_compile": True}),
  ]
