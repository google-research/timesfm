# Window size for decomposed forecasting in TimesFM 2.5

**Date:** 2026-09-03
**Author:** contributor
**TimesFM version:** 2.5
**Status:** approved

## Context

The `window_size` parameter in `src/timesfm/configs.py::ForecastConfig` has been declared since 2025 but never implemented (`TODO(siriuz42):implement it`). It is intended to improve long-term forecast quality by decomposing a time series into trend and residual components.

This feature already existed in TimesFM 1.0 (`v1/src/timesfm/timesfm_jax.py::forecast`) but was lost during the 2.5 migration.

## How it worked in TimesFM 1.0

1. `moving_average(arr, window_size)` split the series into:
   - `smoothed_arr` — moving average (trend)
   - `arr - smoothed_arr` — residual
2. Both components (and optionally the raw series) were fed to the model
3. The model produced a forecast for each component
4. Trend and residual forecasts were summed: `forecast(trend) + forecast(residual) ≈ forecast(raw)`

## Solution for TimesFM 2.5

### Approach

Decomposition into trend and residual with summation of forecasts on output, without duplicating the raw series (reduces inference time compared to v1).

### Architectural rationale

TimesFM 2.5 is a univariate model. Batch is used only for parallelism, there is no cross-series attention. Therefore there is no quality difference between feeding trend and residual in a single batch or in separate calls. Combining into a single call is preferable for reduced overhead.

### Modified files

| File | Change |
|------|--------|
| `src/timesfm/timesfm_2p5/timesfm_2p5_base.py` | Added `moving_average()`. Modified `forecast()` and `forecast_with_covariates()`. |
| `tests/test_base_utils.py` | Added tests for `moving_average()`. |
| `tests/test_model_loading.py` | Added integration tests for `window_size` in `forecast()`. |
| `src/timesfm/configs.py` | Removed TODO, added full parameter description. |

### Algorithm

```
forecast(inputs, horizon):
  if window_size > 0:
    for each ts in inputs:
      trend = moving_average(ts, window_size)
      residual = ts - trend
      expanded_inputs += [trend, residual]
    inputs = expanded_inputs

  # Standard forecast (batching, padding, decode)
  point, quantiles = compiled_decode(horizon, inputs, masks)

  if window_size > 0:
    # Forecasts come in pairs: [trend_0, residual_0, trend_1, residual_1, ...]
    point = sum pairwise (point[0::2] + point[1::2])
    quantiles = sum pairwise (quantiles[0::2] + quantiles[1::2])

  return point[:num_original], quantiles[:num_original]
```

### Edge cases

- `window_size = 0` (default): behavior unchanged — backward compatible.
- `window_size >= len(ts)`: moving average uses actual array length, decomposition degenerates.
- `window_size = 1`: moving average = original series, residual = 0.
- Empty array: returns two empty arrays.

### What we do not cover

- Decomposition type — only MA (moving average), like in v1.
- `forecast_with_covariates()` is blocked with an explicit error when `window_size > 0`.

### Testing

1. **Unit tests for `moving_average()`**: 9 tests — constant series, sum equals original, smoothing, window_size=1, oversized window, empty array, dtype, concrete MA values.
2. **Integration tests**: create `TimesFM_2p5_200M_torch`, compile with `window_size > 0`, verify output shape and covariate error.
