# Trading Bot Slash Commands Guide

This repo includes custom Claude Code slash commands under `.claude/commands/`. They automate common workflows (setup, checks, backtests, training, model inspection) with repo-specific safety rules.

## How slash commands work

- Each file in `.claude/commands/` becomes a slash command.
  - Example: `.claude/commands/tb.check.md` → `/tb.check`
- Anything you type after the command becomes `$ARGUMENTS` inside that command template.
- Claude Code may run terminal commands as part of a slash command (depending on your permissions and prompts).

## How to use

1. Open the repo root in Claude Code (the folder that contains `.claude/commands/`).
2. Run `/tb.help` to see all available commands.
3. Run a command with optional args, e.g.:
   - `/tb.check --fix`
   - `/tb.backtest --strategy atr_breakout --ticker BTCUSDT --interval 1h --period 1y --run`

If commands don’t appear, restart/reopen Claude Code so it reloads `.claude/commands/`.

## Safety conventions used here

- Commands that are network/disk heavy or long-running require an explicit flag:
  - `--run` for long-running or networked actions (paper trading loop, cache warming, metrics server).
  - `--yes` for destructive actions (model purge).
- Commands avoid printing secret values (notably `/tb.env_check`).

## Quick workflows

- First time setup:
  - `/tb.setup`
  - `/tb.check`
- Backtest and read results:
  - `/tb.backtest --strategy atr_breakout --ticker BTCUSDT --interval 1h --period 1y --run`
  - `/tb.report_latest backtest`
- Train an ML strategy and inspect artifacts:
  - `/tb.train --strategy day_trading_ml --ticker BTCUSDT --interval 1h --period 1y --data-source binance --run`
  - `/tb.model_list day_trading_ml__`
- Multi-strategy paper trading (safe default is “print only”):
  - `/tb.paper --strategies long_term,mid_term,atr_breakout`
  - `/tb.paper --strategies long_term,mid_term,atr_breakout --run`

## Command reference

- `/tb.help` — list commands and one-line descriptions.
- `/tb.setup` — create `.venv`, install dependencies, and make the src-layout importable (`pip install -e .`).
- `/tb.check` — run `ruff/black/mypy/pytest` in the repo’s preferred order and summarize failures.
- `/tb.env_check` — compare `trading-bot/.env` vs `trading-bot/.env.example` (missing/empty keys; never prints values).
- `/tb.config_show` — show the effective validated config for a strategy (after runtime defaults) and its persistence key.
- `/tb.paper` — prepare (and only run when asked) a paper-trading command (single or multi-strategy).
- `/tb.backtest` — run the built-in backtest CLI (technical strategies) and summarize the newest report.
- `/tb.train` — train an ML strategy and summarize the newest training report and persisted artifacts.
- `/tb.cache_warm` — prefetch OHLCV and write cache files (requires `--run`).
- `/tb.report_latest` — summarize the newest JSON under `trading-bot/reports/` (optionally filter `train`/`backtest`).
- `/tb.model_list` — list keys under `trading-bot/saved_models/` and summarize latest metadata.
- `/tb.model_purge` — delete artifacts for a persistence key (requires `--yes`).
- `/tb.lookahead_audit` — run look-ahead focused tests and scan for common “future leak” patterns.
- `/tb.metrics_server` — start a local Prometheus `/metrics` endpoint (requires `--run`).
- `/tb.add_strategy` — checklist-style workflow to add a new strategy end-to-end (config/impl/registry/validation/tests).
- `/tb.add_indicator` — add a new indicator module + tests + config validation entry.
- `/tb.debug_test` — reproduce and fix a failing test with a minimal regression test.

## Commits & milestones (recommended)

- Commit small, coherent changes and keep commits green (run `/tb.check` before committing).
- Commit at natural milestones: bug fix + regression test, refactor completion, new strategy wired + validated, backtest change + updated expectations.
- Use consistent messages (e.g., Conventional Commits) and include “why” + reproduction commands for trading-logic/data changes.

## Sharing with a team (optional)

You can commit `.claude/commands/` and `commands-help.md` to share them with a team. Consider keeping `.claude/settings.local.json` local or machine-specific (it controls which shell commands Claude Code can auto-run).
