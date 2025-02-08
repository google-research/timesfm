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
"""PyTorch dataset adapter for TimesFM.

This dataset is designed to be used with `timesfm.data_loader.TimeSeriesdata`.
"""

from typing import Callable, Generator, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


class TimesFmIterableDataset(IterableDataset):
  """PyTorch dataset adapter for TimesFM.

  Args:
    generator_fn: A function that returns a generator that yields tuples of
      numpy arrays. For the expected format of the tuples, see `timesfm.data_loader.TimeSeriesdata.train_gen()`.
  """
  def __init__(
    self,
    generator_fn: Callable[
      [],
      Generator[
        Tuple[
          np.float32,
          np.float32,
          np.int32,
          np.float32,
          np.float32,
          np.int32,
          np.int32,
        ],
        None,
        None,
      ],
    ],
  ):
    super().__init__()
    self._generator_fn = generator_fn

  def _generator_wrapper(self, worker_id: int, num_workers: int):
    """Wraps the generator to yield data only for the current worker."""
    i = 0
    for data in self._generator_fn():
      if i % num_workers == worker_id:
        yield data
      i += 1

  def __iter__(self):
    """Returns an iterator over the dataset."""
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
      return self._generator_fn()

    worker_id = worker_info.id
    num_workers = worker_info.num_workers
    return self._generator_wrapper(worker_id, num_workers)
