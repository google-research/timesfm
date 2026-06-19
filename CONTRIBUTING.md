# How to Contribute

We would love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Development setup

See [development_setup.md](development_setup.md) for environment setup, running
tests, and the fork → PR workflow.

## Contribution process

### Code Reviews

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.

### Where to contribute

- **TimesFM 2.5 (current):** `src/timesfm/` and `tests/`
- **Examples and docs:** `timesfm-forecasting/`, `README.md`
- **Archived v1/v2 models:** `v1/` (legacy; separate Poetry environment)

### Pre-submit checklist

```bash
pytest tests/ --ignore=tests/test_model_loading.py
ruff check src/ tests/
uv run python -m build
```
