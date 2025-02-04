from typing import Callable, Generator, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


class TimesFmIterableDataset(IterableDataset):
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
    i = 0
    for data in self._generator_fn():
      if i % num_workers == worker_id:
        yield data
      i += 1

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
      return self._generator_fn()

    worker_id = worker_info.id
    num_workers = worker_info.num_workers
    return self._generator_wrapper(worker_id, num_workers)
