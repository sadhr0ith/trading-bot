---
description: Set up the local Python environment for this repo (venv + deps + editable install) and verify imports.
---

## User Input

```text
$ARGUMENTS
```

## Goal

Get to a working state where these run successfully from `trading-bot/`:

- `python -m trading_bot.main --help`
- `pytest`
- `ruff check .`

## Flow

1. Work from the package root:
   - `cd trading-bot`
2. Create venv if missing:
   - `python3 -m venv .venv`
3. Activate venv:
   - `source .venv/bin/activate`
4. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -r requirements-dev.txt`
5. Ensure the src-layout package is importable (recommended):
   - `pip install -e .`
   - If editable install fails, fallback: `export PYTHONPATH=src`
6. Create `.env` if missing (do not overwrite an existing one):
   - `cp .env.example .env`
7. Verify:
   - `python -c "import trading_bot; print(trading_bot.__file__)"`
   - `python -m trading_bot.main --help`

## Output requirements

- Print the commands you ran and whether each step succeeded.
- If something fails, stop and report the exact error + the next smallest fix.

