# Development Setup

Guide for contributing to TimesFM 2.5. Archived v1/v2 code in `v1/` uses a
separate Poetry environment — see `v1/README.md` if you need legacy models.

## Prerequisites

- Python 3.10+ (3.11 recommended; matches CI)
- [uv](https://docs.astral.sh/uv/) package manager

## Local environment

```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm

uv python install 3.11
uv python pin 3.11
uv venv --python 3.11
source .venv/bin/activate

# PyTorch backend (recommended for most contributors)
uv pip install -e ".[dev,torch]"

# Or install all backends for full test coverage
uv pip install -e ".[dev,torch,flax]"
```

On Apple Silicon or GPU machines, you may need to reinstall torch/jax from the
official install guides after the editable install.

## Verify setup

```bash
python -c "import timesfm; print(timesfm.TimesFM_2p5_200M_torch)"
pytest tests/ --ignore=tests/test_model_loading.py
uv run python -m build
```

## Fork and pull request workflow

1. **Fork** [google-research/timesfm](https://github.com/google-research/timesfm)
   on GitHub.

2. **Add your fork** as a remote:

   ```bash
   git remote add fork https://github.com/<your-username>/timesfm.git
   ```

3. **Sign the Google CLA** at <https://cla.developers.google.com/> (required
   before your PR can be merged).

4. **Create a branch**, make changes, and open a PR:

   ```bash
   git checkout -b feature/my-change
   pytest tests/ --ignore=tests/test_model_loading.py
   ruff check src/ tests/
   git add <files>
   git commit -m "Describe why, not just what"
   git push -u fork feature/my-change
   ```

   Open a pull request against `google-research/timesfm:master`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for community guidelines and review
process.

## Test matrix

| Test file | Requires |
|-----------|----------|
| `tests/test_configs.py` | core |
| `tests/test_base_utils.py` | core |
| `tests/test_torch_layers.py` | `[torch]` |
| `tests/test_torch_utils.py` | `[torch]` |
| `tests/test_model_loading.py` | `[torch]` + `[flax]` |

## Lint

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```
