# Copyright 2024 The Google Research Authors.
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
"""Regression tests for v1 TimeSeriesdata batching."""

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pandas as pd


_V1_TIMESFM_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "timesfm"


def _install_optional_dependency_stubs():
  """Installs lightweight stubs for optional deps needed in module imports."""
  if "absl" not in sys.modules:
    absl_module = types.ModuleType("absl")
    logging_module = types.ModuleType("absl.logging")
    logging_module.info = lambda *args, **kwargs: None
    absl_module.logging = logging_module
    sys.modules["absl"] = absl_module
    sys.modules["absl.logging"] = logging_module

  if "sklearn" not in sys.modules:
    sklearn_module = types.ModuleType("sklearn")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")

    class _IdentityScaler:

      def fit(self, matrix):
        return self

      def transform(self, matrix):
        return matrix

      def fit_transform(self, matrix):
        return matrix

    preprocessing_module.StandardScaler = _IdentityScaler
    sklearn_module.preprocessing = preprocessing_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.preprocessing"] = preprocessing_module

  if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = types.ModuleType("tensorflow")

  if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterator, *args, **kwargs: iterator
    sys.modules["tqdm"] = tqdm_module


def _load_time_series_data():
  """Loads TimeSeriesdata without importing the package __init__."""
  package_name = "timesfm_data_loader_testpkg"
  module_name = f"{package_name}.data_loader"
  if module_name in sys.modules:
    return sys.modules[module_name].TimeSeriesdata

  _install_optional_dependency_stubs()

  package_module = types.ModuleType(package_name)
  package_module.__path__ = [str(_V1_TIMESFM_SRC)]
  sys.modules[package_name] = package_module

  for submodule in ("time_features", "data_loader"):
    full_name = f"{package_name}.{submodule}"
    module_path = _V1_TIMESFM_SRC / f"{submodule}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    loaded_module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = loaded_module
    spec.loader.exec_module(loaded_module)

  return sys.modules[module_name].TimeSeriesdata


def _build_loader(tmp_path: pathlib.Path, num_ts: int, batch_size: int):
  """Creates a minimal TimeSeriesdata instance for train_gen tests."""
  csv_path = tmp_path / "timeseries.csv"
  num_rows = 40
  date_range = pd.date_range("2025-01-01", periods=num_rows, freq="D")

  frame = pd.DataFrame({"datetime": date_range})
  ts_cols = []
  for idx in range(num_ts):
    column_name = f"ts_{idx}"
    frame[column_name] = np.arange(num_rows) + idx
    ts_cols.append(column_name)
  frame.to_csv(csv_path, index=False)

  loader_cls = _load_time_series_data()
  return loader_cls(
      data_path=csv_path,
      datetime_col="datetime",
      num_cov_cols=[],
      cat_cov_cols=[],
      ts_cols=ts_cols,
      train_range=(0, 12),
      val_range=(12, 24),
      test_range=(24, 40),
      hist_len=8,
      pred_len=2,
      batch_size=batch_size,
      freq="D",
      normalize=False,
      epoch_len=1,
      permute=False,
  )


def test_train_gen_non_permute_respects_batch_windows(tmp_path):
  loader = _build_loader(tmp_path, num_ts=5, batch_size=2)

  batches = list(loader.train_gen())
  assert [batch[-1].tolist() for batch in batches] == [[0, 1], [2, 3], [4]]

  for batch in batches:
    bts_train = batch[0]
    bts_pred = batch[3]
    tsidx = batch[-1]
    assert bts_train.shape[0] == len(tsidx)
    assert bts_pred.shape[0] == len(tsidx)


def test_train_gen_non_permute_no_extra_batches_on_even_split(tmp_path):
  loader = _build_loader(tmp_path, num_ts=4, batch_size=2)

  batches = list(loader.train_gen())
  assert len(batches) == 2
  assert [batch[-1].tolist() for batch in batches] == [[0, 1], [2, 3]]
