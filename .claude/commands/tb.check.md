---
description: Run the local quality gates (format, lint, types, tests) for this repo and summarize failures with next fixes.
---

## User Input

```text
$ARGUMENTS
```

## What this command does

1. Work from `trading-bot/` (the Python package root). If needed, change directory first.
2. Run the quality gates in this order:
   - `ruff check .`
   - `ruff format .`
   - `black .`
   - `mypy`
   - `pytest`
3. If `$ARGUMENTS` contains `fix` or `--fix`, run `ruff check --fix .` before re-running `ruff check .`.
4. If any step fails:
   - Do not continue blindly; capture the error output.
   - Identify the smallest fix that unblocks the next gate.
   - Re-run only the failing step(s) until green.
5. Report back with:
   - Which commands were run
   - Which step failed (if any) and why
   - The exact file(s) changed to fix it
   - The final status (all green or remaining failures)

## Repo specifics (must follow)

- Keep diffs focused; do not do drive-by refactors.
- Tests should stay offline/deterministic (prefer synthetic data + monkeypatch).
