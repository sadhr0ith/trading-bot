---
description: Train an ML strategy (separate from inference), then summarize the generated training report and persisted artifacts.
---

## User Input

```text
$ARGUMENTS
```

## Preconditions

- Training is for ML strategies (e.g. `day_trading_ml`, `day_trading`, `short_term`, `mid_term`, `long_term`).
- Training may require network access for data fetching depending on `data_source`.

## Flow

1. Work from `trading-bot/`.
2. Parse `$ARGUMENTS` for:
   - `--strategy` (required)
   - Optional overrides: `--ticker`, `--interval`, `--period`, `--data-source`
   - Optional: `--report-dir` (default `reports`)
3. Build the training command:
   - `python -m trading_bot.train --strategy <strategy> --report-dir reports`
   - Include any provided overrides.
4. If `$ARGUMENTS` contains `run` or `--run`, execute the command; otherwise print the command and ask for confirmation.
5. After training:
   - Find the newest `reports/train_*.json`
   - Summarize: `config_signature`, `dataset_hash`, feature columns/version, and key metrics written by persistence
   - Confirm artifacts exist under `saved_models/<persistence_key>/` for the trained context

## Output requirements

- Print the exact command used (or proposed).
- Print the report path and a concise metrics summary.
- Call out if metrics are missing (e.g., no metadata found) or if `feature_shift`/data health looks bad.
