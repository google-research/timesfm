# TimesFM — Agent Entry Point

This repository ships a first-party **Agent Skill** for TimesFM at:

```
timesfm-forecasting/
└── SKILL.md    ← read this for the full skill
```

## Install the skill

Copy the skill directory into your agent's skills folder:

```bash
# Cursor / Claude Code / OpenCode / Codex (global install)
cp -r timesfm-forecasting/ ~/.cursor/skills/
cp -r timesfm-forecasting/ ~/.claude/skills/

# Or project-level
cp -r timesfm-forecasting/ .cursor/skills/
```

Any agent that supports the open [Agent Skills standard](https://agentskills.io) will discover it automatically.

## Working in this repo

If you are developing TimesFM itself (not using it), the source lives in:

- `src/timesfm3/` — TimesFM 3.0 (PyTorch backend in `torch/`, MLX backend in `mlx/`)
- `src/timesfm/` — TimesFM 2.5 (PyTorch and Flax)

Archived v1/v2 code and notebooks are in `v1/`.

Run tests:

```bash
pip install -e ".[torch]" pytest

# TimesFM 3.0 (MLX tests are skipped unless `mlx` is installed)
pytest src/timesfm3/

# TimesFM 2.5 (tests/test_model_loading.py also needs the `flax` extra)
pytest tests/ --ignore=tests/test_model_loading.py
```

Tests for the archived v1/v2 code are in `v1/tests/`.

See `README.md` for full developer setup.
