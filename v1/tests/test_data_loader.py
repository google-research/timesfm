from pathlib import Path

import numpy as np
import pandas as pd

from timesfm.data_loader import TimeSeriesdata


def test_train_gen_respects_batch_size_when_permute_is_false(tmp_path: Path) -> None:
    rows = 12
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "ts_1": np.arange(rows),
            "ts_2": np.arange(rows) + 10,
            "ts_3": np.arange(rows) + 20,
            "ts_4": np.arange(rows) + 30,
            "ts_5": np.arange(rows) + 40,
        }
    )
    data_path = tmp_path / "sample.csv"
    df.to_csv(data_path, index=False)

    loader = TimeSeriesdata(
        data_path=str(data_path),
        datetime_col="ds",
        num_cov_cols=None,
        cat_cov_cols=None,
        ts_cols=np.array(["ts_1", "ts_2", "ts_3", "ts_4", "ts_5"]),
        train_range=[0, 8],
        val_range=[8, 10],
        test_range=[10, 12],
        hist_len=3,
        pred_len=2,
        batch_size=2,
        freq="D",
        normalize=False,
        epoch_len=1,
        holiday=False,
        permute=False,
    )

    batches = list(loader.train_gen())
    ts_indices = [batch[-1].tolist() for batch in batches]

    assert ts_indices == [[0, 1], [2, 3], [4]]
    for batch in batches:
        assert len(batch[-1]) <= 2
        assert batch[0].shape[0] == len(batch[-1])
        assert batch[3].shape[0] == len(batch[-1])
