# Copyright 2024 Google LLC
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
"""TimesFM init file."""
print(
    "TimesFM v1.2.0. See https://github.com/google-research/timesfm/blob/master/README.md for updated APIs."
)
from timesfm.timesfm_base import freq_map, TimesFmCheckpoint, TimesFmHparams, TimesFmBase
try:
  print("Loaded Jax TimesFM.")
  from timesfm.timesfm_jax import TimesFmJax as TimesFm
  from timesfm import data_loader
except Exception as _:
  print("Loaded PyTorch TimesFM.")
  from timesfm.timesfm_torch import TimesFmTorch as TimesFm
