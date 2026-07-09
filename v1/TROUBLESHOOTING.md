# Troubleshooting

This document provides solutions to common issues encountered when using TimesFM.

## Installation Issues

### ARM/Apple Silicon Compatibility
**Problem:** `lingvo` dependency fails on Apple Silicon (M1/M2/M3) machines.
```
ERROR: Could not build wheels for lingvo
```
**Solution:** This is a known issue. The `lingvo` dependency doesn't support ARM architectures. We recommend:
- Use x86_64 emulation via Rosetta 2: `arch -x86_64 pip install timesfm[pax]`
- Use the PyTorch version instead, which has better ARM support: `pip install timesfm[torch]`
- Use Docker with x86_64 emulation for consistent environments

### Memory Issues During Installation
**Problem:** Installation fails with memory errors.
```
Killed (signal 9)
```
**Solution:** 
- Ensure at least 32GB RAM is available
- Close other applications during installation
- Use `pip install --no-cache-dir timesfm[torch]` to reduce memory usage
- Install in a clean virtual environment

### JAX/PyTorch Version Conflicts
**Problem:** Conflicting JAX and PyTorch installations.
```
ImportError: cannot import name 'jax' from 'jax'
```
**Solution:**
- For PyTorch-only usage: `pip install timesfm[torch]`
- For covariates with PyTorch: `pip install timesfm[torch] && pip install jax jaxlib`
- For PAX version: `pip install timesfm[pax]`

## Runtime Errors

### Model Loading Issues
**Problem:** Checkpoint download fails or is corrupted.
```
HfFileNotFoundError: 404 Client Error
```
**Solution:**
- Check internet connectivity
- Verify Hugging Face Hub access: `huggingface-cli login`
- Clear cache: `rm -rf ~/.cache/huggingface/`
- Use explicit checkpoint paths if needed

### CUDA/GPU Issues
**Problem:** GPU not detected or CUDA errors.
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `per_core_batch_size` (try 16, 8, or 4)
- Reduce `context_len` to minimum needed
- Use `backend="cpu"` for testing
- Check GPU memory: `nvidia-smi`

### Context Length Errors
**Problem:** Input series longer than model capacity.
```
ValueError: context_len must be <= 512 for v1.0 models
```
**Solutions:**
- Use TimesFM-2.0 for longer contexts (up to 2048)
- Ensure `context_len` is multiple of 32
- Truncate input series if necessary
- Set appropriate `context_len` in model initialization

## Data Issues

### Frequency Mapping Problems
**Problem:** Unexpected forecasting results with wrong frequency.
```
Warning: Frequency 'D' mapped to category 0
```
**Solutions:**
- Verify frequency mapping: D→0 (high), W/M→1 (medium), Q/Y→2 (low)
- Override automatic mapping by specifying frequency manually
- Check data granularity matches chosen frequency category

### Missing Values in Time Series
**Problem:** NaN or missing values in input data.
```
ValueError: Input contains NaN values
```
**Solutions:**
- Pre-process data to handle missing values (forward fill, interpolation)
- Ensure continuous time series without gaps
- Remove or impute missing values before forecasting

### Covariate Dimension Mismatches
**Problem:** Covariate lengths don't match forecast horizon.
```
ValueError: Dynamic covariates must cover context + horizon
```
**Solutions:**
- Ensure dynamic covariates have length = context + horizon
- Check static vs dynamic covariate classification
- Verify covariate data alignment with time series

## Performance Issues

### Slow Inference
**Problem:** Forecasting takes unexpectedly long.
**Solutions:**
- Use GPU backend: `backend="gpu"`
- Optimize batch size: increase `per_core_batch_size`
- Use appropriate model size for your use case
- Profile with smaller data first

### Memory Usage
**Problem:** High memory consumption during inference.
**Solutions:**
- Reduce batch size: `per_core_batch_size=1`
- Process data in chunks
- Use smaller context length when possible
- Monitor memory with `htop` or `nvidia-smi`

## Common Error Messages

### `ModuleNotFoundError: No module named 'xreg_lib'`
**Cause:** Missing JAX dependencies for covariates functionality.
**Solution:** `pip install jax jaxlib`

### `ValueError: horizon_len must be positive`
**Cause:** Invalid horizon length specified.
**Solution:** Set `horizon_len > 0` in model initialization.

### `RuntimeError: Expected input batch_size (X) to be divisible by batch_size (Y)`
**Cause:** Batch size mismatch.
**Solution:** Adjust `per_core_batch_size` or input data batching.

## Getting Help

If you encounter issues not covered here:
1. Check the [GitHub Issues](https://github.com/google-research/timesfm/issues)
2. Review the [notebooks/](notebooks/) for working examples
3. Verify your installation follows the exact steps in the Installation section
4. Test with the provided example data before using your own datasets