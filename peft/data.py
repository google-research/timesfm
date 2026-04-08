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

"""Time-series dataset for fine-tuning TimesFM 2.5."""

import math
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
  """Sliding-window dataset that produces (context, mask, target) tuples.

  Accepts data in several formats:

  * **list of arrays** — each element is a 1-D NumPy array or Python list
    representing a single time series.
  * **long-format DataFrame** — columns ``[id_col, value_col]`` where each
    unique ``id_col`` identifies a series.
  * **wide-format DataFrame** — every numeric column is treated as an
    independent time series.

  For each series the dataset generates sliding windows of length
  ``context_len + horizon_len`` with the given ``stride``.  Series shorter
  than the window are left-padded with zeros and masked.

  Args:
    data: Time-series data (see above).
    context_len: Context (input) length.  Will be rounded up to a multiple
      of ``patch_len`` (32).
    horizon_len: Prediction horizon.  Must be ≤ 128.
    stride: Step size between consecutive windows.
    patch_len: Patch size used by the model (default 32).
    id_col: Column name for series identifier (long-format DataFrames).
    value_col: Column name for values (long-format DataFrames).
  """

  PATCH_LEN = 32
  MAX_HORIZON = 128

  def __init__(
    self,
    data: Union[List[np.ndarray], pd.DataFrame],
    context_len: int = 512,
    horizon_len: int = 128,
    stride: int = 1,
    patch_len: int = PATCH_LEN,
    id_col: Optional[str] = None,
    value_col: Optional[str] = None,
  ):
    if horizon_len > self.MAX_HORIZON:
      raise ValueError(
        f"horizon_len={horizon_len} exceeds the single-step maximum of "
        f"{self.MAX_HORIZON}.  Use a shorter horizon for fine-tuning; the "
        f"model handles longer horizons via autoregressive decoding at "
        f"inference time."
      )

    self.patch_len = patch_len
    # Round context_len up to a multiple of patch_len.
    self.context_len = math.ceil(context_len / patch_len) * patch_len
    self.horizon_len = horizon_len
    self.window_len = self.context_len + horizon_len

    self.series: List[np.ndarray] = self._parse_data(data, id_col, value_col)
    self.windows = self._build_windows(stride)

  # -- Data parsing --------------------------------------------------------

  @staticmethod
  def _parse_data(
    data: Union[List[np.ndarray], pd.DataFrame],
    id_col: Optional[str],
    value_col: Optional[str],
  ) -> List[np.ndarray]:
    if isinstance(data, pd.DataFrame):
      if id_col is not None and value_col is not None:
        # Long format.
        return [
          grp[value_col].to_numpy(dtype=np.float32)
          for _, grp in data.groupby(id_col, sort=False)
        ]
      # Wide format — each numeric column is a series.
      return [
        data[c].to_numpy(dtype=np.float32)
        for c in data.select_dtypes(include="number").columns
      ]
    # List / sequence of arrays.
    return [np.asarray(s, dtype=np.float32) for s in data]

  def _build_windows(self, stride: int) -> List[tuple]:
    windows = []
    for sidx, series in enumerate(self.series):
      slen = len(series)
      if slen < self.window_len:
        # Single (padded) window.
        windows.append((sidx, 0, slen))
      else:
        for start in range(0, slen - self.window_len + 1, stride):
          windows.append((sidx, start, start + self.window_len))
    return windows

  # -- torch Dataset interface ---------------------------------------------

  def __len__(self) -> int:
    return len(self.windows)

  def __getitem__(self, idx: int):
    sidx, start, end = self.windows[idx]
    raw = self.series[sidx][start:end]

    if len(raw) < self.window_len:
      # Left-pad context; target uses whatever tail is available.
      available_ctx = max(0, len(raw) - self.horizon_len)
      target = raw[available_ctx:].copy()
      if len(target) < self.horizon_len:
        target = np.pad(target, (0, self.horizon_len - len(target)))

      ctx_raw = raw[:available_ctx]
      pad_len = self.context_len - len(ctx_raw)
      context = np.pad(ctx_raw, (pad_len, 0)).astype(np.float32)
      mask = np.zeros(self.context_len, dtype=bool)
      mask[:pad_len] = True
    else:
      context = raw[: self.context_len].astype(np.float32)
      mask = np.zeros(self.context_len, dtype=bool)
      target = raw[self.context_len : self.context_len + self.horizon_len].astype(
        np.float32
      )

    return (
      torch.from_numpy(context),
      torch.from_numpy(mask),
      torch.from_numpy(target),
    )
