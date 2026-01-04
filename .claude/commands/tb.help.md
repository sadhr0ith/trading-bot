---
description: Show the available trading-bot slash commands and what each one does.
---

## Available commands

- `/tb.setup` — create venv, install deps, ensure editable install works
- `/tb.check` — run format/lint/types/tests and summarize failures
- `/tb.env_check` — compare `.env` vs `.env.example` without printing secrets
- `/tb.config_show` — print effective validated strategy config + persistence key
- `/tb.paper` — prepare (and optionally run) a paper-trading command safely
- `/tb.backtest` — run backtest CLI and summarize the newest report
- `/tb.train` — train an ML strategy and summarize the newest training report
- `/tb.cache_warm` — prefetch OHLCV and write cache files (requires `--run`)
- `/tb.report_latest` — summarize newest JSON report under `reports/`
- `/tb.model_list` — list saved model keys + latest metadata summary
- `/tb.model_purge` — delete artifacts for a persistence key (requires `--yes`)
- `/tb.lookahead_audit` — run look-ahead focused tests + quick scan
- `/tb.metrics_server` — start local `/metrics` server (requires `--run`)
- `/tb.add_strategy` — scaffold a new strategy end-to-end (config/impl/registry/tests)
- `/tb.add_indicator` — add a new indicator + tests + config validation
- `/tb.debug_test` — reproduce and fix a failing test with a minimal regression test
