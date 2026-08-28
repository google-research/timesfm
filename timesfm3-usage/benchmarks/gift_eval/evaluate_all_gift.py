"""Official Full GIFT-Eval Benchmark Evaluator for TimesFM 3.0 (All 97 Configurations)."""

import os
import sys
import csv
import json
import math
import logging
import datetime
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The mean prediction is not stored in the forecast data.*")

def find_repo_root():
    curr = Path.cwd().resolve()
    for p in [curr] + list(curr.parents):
        if (p / "src" / "timesfm3").exists() or (p / "pyproject.toml").exists():
            return p
    return curr


repo_root = find_repo_root()
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

default_gift_dir = os.path.expanduser("~/data/gift_eval")
os.environ["GIFT_EVAL"] = os.getenv("GIFT_EVAL", default_gift_dir)

# Set up results directory & logging
results_dir = repo_root / "results/TimesFM-3_multivariate"
results_dir.mkdir(parents=True, exist_ok=True)

log_file_path = results_dir / "evaluation.log"


class SuppressGluonTSWarningFilter(logging.Filter):
    """Filters out noisy GluonTS QuantileForecast mean-to-median fallback warnings."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "The mean prediction is not stored in the forecast data" in msg:
            return False
        return True


filter_instance = SuppressGluonTSWarningFilter()
file_handler = logging.FileHandler(str(log_file_path), mode="a")
file_handler.addFilter(filter_instance)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.addFilter(filter_instance)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[file_handler, stream_handler],
)
logging.getLogger("gluonts").setLevel(logging.ERROR)
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)
logging.getLogger().addFilter(filter_instance)
logger = logging.getLogger("TimesFM3-GIFT-Eval")

from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.itertools import batcher
from gluonts.model import evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality
from gift_eval.data import Dataset, Term

from timesfm3 import TimesFM3Evaluator, ModelConfig

# Standard GIFT-Eval evaluation metrics
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

# Official dataset list from gift-eval timesfm2p5.ipynb and gift_eval_timesfm3.ipynb
short_datasets = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W hospital covid_deaths "
    "us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W "
    "ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
)

med_long_datasets = (
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H "
    "ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

all_datasets = sorted(list(set(short_datasets.split() + med_long_datasets.split())))

# Load metadata properties mapping
props_path = repo_root / "gift-eval/notebooks/dataset_properties.json"
with open(props_path, "r") as f:
    dataset_properties_map = json.load(f)

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
    "loop_seattle": "loop_seattle",
    "m_dense": "m_dense",
    "sz_taxi": "sz_taxi",
}


class TimesFm3Predictor:
    """GluonTS Predictor wrapper for TimesFM 3.0 with adaptive batching."""

    def __init__(self, forecaster: TimesFM3Evaluator, prediction_length: int, batch_size: int = 64):
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def predict(self, test_data_input, batch_size: int = None) -> List[QuantileForecast]:
        if batch_size is None:
            batch_size = self.batch_size
        self.forecaster.config.per_core_batch_size = batch_size

        if hasattr(test_data_input, "__len__"):
            inputs_list = test_data_input
            total_series = len(test_data_input)
        else:
            inputs_list = list(test_data_input)
            total_series = len(inputs_list)

        total_batches = math.ceil(total_series / batch_size) if total_series > 0 else 0
        forecasts = []
        batch_idx = 0

        for batch in batcher(inputs_list, batch_size=batch_size):
            contexts = [np.array(entry["target"], dtype=np.float32) for entry in batch]
            past_covs = [
                np.array(entry["past_feat_dynamic_real"], dtype=np.float32)
                if ("past_feat_dynamic_real" in entry and entry["past_feat_dynamic_real"] is not None)
                else None
                for entry in batch
            ]
            outputs = list(
                self.forecaster.predict_batch(
                    contexts=contexts,
                    horizon=self.prediction_length,
                    past_only_covariates=past_covs,
                    return_quantiles=True,
                    use_symmetric_averaging=True,
                    make_positive=True,
                    sort_quantiles=True,
                )
            )
            for out, entry in zip(outputs, batch):
                target_len = entry["target"].shape[-1]
                forecast_start_date = entry["start"] + target_len
                if out.quantiles.ndim == 3:
                    q_arr = np.transpose(out.quantiles[:, : self.prediction_length, :], (2, 1, 0))
                else:
                    q_arr = out.quantiles[: self.prediction_length, :].T
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=q_arr,
                        forecast_keys=[str(q) for q in self.quantiles],
                        start_date=forecast_start_date,
                    )
                )
            batch_idx += 1
            if batch_idx % 10 == 0 or batch_idx == total_batches or total_batches <= 10:
                logger.info(f"  [Predictor] Batch {batch_idx}/{total_batches} complete ({len(forecasts)}/{total_series} forecasts generated)...")
        return forecasts


def geomean(vals):
    arr = np.array(vals, dtype=np.float64)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    return float(np.exp(np.mean(np.log(valid)))) if len(valid) > 0 else np.nan


def main():
    logger.info("=" * 80)
    logger.info("Starting TimesFM-3.0 Official GIFT-Eval Benchmark Evaluation")
    logger.info("=" * 80)
    logger.info(f"PyTorch version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Active GPU: {torch.cuda.get_device_name(0)}")

    config = ModelConfig(
        checkpoint_path=os.getenv(
            "TIMESFM3_CHECKPOINT", "google/timesfm-3.0-pytorch"
        ),
        input_patch_length=32,
        output_patch_length=64,
        per_core_batch_size=64,
        use_variate_attention=True,
        use_sdpa=True,
    )
    forecaster = TimesFM3Evaluator(config)
    logger.info(f"Loaded TimesFM-3 Evaluator on: {forecaster.device}")

    model_name = "TimesFM-3"
    csv_file_path = results_dir / "all_results.csv"

    header = [
        "dataset",
        "model",
        "eval_metrics/MSE[mean]",
        "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain",
        "num_variates",
    ]

    completed_configs = set()
    if csv_file_path.exists() and csv_file_path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(csv_file_path)
            completed_configs.update(existing_df["dataset"].dropna().tolist())
        except Exception:
            pass

    if not csv_file_path.exists() or csv_file_path.stat().st_size == 0:
        with open(csv_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    logger.info(f"Resume Status: Found {len(completed_configs)} already completed configurations.")

    total_configs_run = 0

    for ds_num, ds_name in enumerate(all_datasets):
        terms = ["short", "medium", "long"]
        for term in terms:
            if (term in ["medium", "long"]) and (ds_name not in med_long_datasets.split()):
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]

            ds_config = f"{ds_key}/{ds_freq}/{term}"
            total_configs_run += 1

            if ds_config in completed_configs:
                logger.info(f"[{total_configs_run}/97] Skipping {ds_config} (already completed)")
                continue

            domain = dataset_properties_map[ds_key]["domain"]
            num_variates = dataset_properties_map[ds_key]["num_variates"]
            batch_size = 16 if num_variates > 1 else 64

            logger.info(f"\n[{total_configs_run}/97] Evaluating {ds_config} (disk: {ds_name}, batch_size={batch_size}, variates={num_variates})...")

            try:
                dataset = Dataset(name=ds_name, term=term, to_univariate=False)
                seasonality = get_seasonality(dataset.freq)
                total_series = len(dataset.test_data)
                total_batches = math.ceil(total_series / batch_size)

                # Inspect first sample for covariate inspection
                sample_item = dataset.test_data[0] if hasattr(dataset.test_data, "__getitem__") else next(iter(dataset.test_data))
                sample = sample_item[0] if isinstance(sample_item, tuple) else sample_item
                tgt_shape = np.array(sample["target"]).shape
                has_po = "past_feat_dynamic_real" in sample and sample["past_feat_dynamic_real"] is not None
                po_dim = np.array(sample["past_feat_dynamic_real"]).shape[0] if has_po else 0

                logger.info(
                    f"  -> Dataset Info: Total series/windows={total_series:,} | Targets={num_variates} (shape {tgt_shape}) | "
                    f"Past covariates={po_dim} | Total batches={total_batches} (batch_size={batch_size})"
                )

                predictor = TimesFm3Predictor(
                    forecaster=forecaster,
                    prediction_length=dataset.prediction_length,
                    batch_size=batch_size,
                )

                res = evaluate_model(
                    predictor,
                    test_data=dataset.test_data,
                    metrics=metrics,
                    batch_size=batch_size,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=seasonality,
                )

                row = [
                    ds_config,
                    model_name,
                    res["MSE[mean]"][0],
                    res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0],
                    res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0],
                    res["sMAPE[0.5]"][0],
                    res["MSIS"][0],
                    res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0],
                    res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    domain,
                    num_variates,
                ]

                with open(csv_file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                completed_configs.add(ds_config)
                logger.info(
                    f"  [{ds_config}] MASE[0.5]: {res['MASE[0.5]'][0]:.4f} | "
                    f"CRPS (MWQL): {res['mean_weighted_sum_quantile_loss'][0]:.4f} | "
                    f"sMAPE: {res['sMAPE[0.5]'][0]:.4f} | NRMSE: {res['NRMSE[mean]'][0]:.4f}"
                )

            except Exception as e:
                logger.error(f"  -> ERROR on {ds_config}: {e}", exc_info=True)

    # Final summary over all configurations
    df = pd.read_csv(csv_file_path)
    df = df.drop_duplicates(subset=["dataset"], keep="last")

    logger.info("\n" + "=" * 75)
    logger.info(f"TOTAL EVALUATED CONFIGURATIONS: {len(df)} / 97")
    logger.info("=" * 75)
    logger.info(f"Overall Geometric Mean MASE:        {geomean(df['eval_metrics/MASE[0.5]']):.6f}")
    logger.info(f"Overall Geometric Mean CRPS (MWQL): {geomean(df['eval_metrics/mean_weighted_sum_quantile_loss']):.6f}")
    logger.info(f"Overall Geometric Mean sMAPE:       {geomean(df['eval_metrics/sMAPE[0.5]']):.6f}")
    logger.info(f"Overall Arithmetic Mean MASE:       {df['eval_metrics/MASE[0.5]'].mean():.6f}")
    logger.info(f"Overall Arithmetic Mean CRPS:       {df['eval_metrics/mean_weighted_sum_quantile_loss'].mean():.6f}")
    logger.info("=" * 75)

    domain_summary = df.groupby("domain").agg(
        count=("dataset", "count"),
        geomean_MASE=("eval_metrics/MASE[0.5]", geomean),
        geomean_CRPS=("eval_metrics/mean_weighted_sum_quantile_loss", geomean),
        mean_MASE=("eval_metrics/MASE[0.5]", "mean"),
        mean_CRPS=("eval_metrics/mean_weighted_sum_quantile_loss", "mean"),
        mean_sMAPE=("eval_metrics/sMAPE[0.5]", "mean"),
    ).reset_index()

    logger.info("\n--- Domain Performance Breakdown ---")
    logger.info("\n" + domain_summary.to_string(index=False))


if __name__ == "__main__":
    main()
