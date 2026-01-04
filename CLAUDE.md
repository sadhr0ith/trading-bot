# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-strategy trading bot (crypto/stocks) with ML pipelines, technical indicators, paper trading, and backtesting.
Paper trading only by default (no real execution). Python 3.12 recommended (3.10+ required).

## Repo Layout

- The Python package lives in `trading-bot/` (src-layout under `trading-bot/src/`).
- Most commands below assume you first run: `cd trading-bot`

## Commands

```bash
cd trading-bot

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Make the src-layout importable (pick one)
pip install -e .
# or:
export PYTHONPATH=src

# Tests (coverage configured in pyproject)
pytest

# Lint / format / types
ruff check .
ruff format .
black .
mypy

# Paper trading (single strategy)
python -m trading_bot.main --strategy atr_breakout

# Paper trading (multi-strategy aggregation)
python -m trading_bot.main --strategies long_term,mid_term,atr_breakout

# ML training (separate from inference)
python -m trading_bot.train --strategy day_trading_ml --report-dir reports

# Backtest (single-asset; technical strategies only)
python -m trading_bot.backtest.cli --strategy atr_breakout --ticker BTCUSDT --interval 1h --period 1y --report-dir reports

# Portfolio backtest
python -m trading_bot.backtest.portfolio_cli --strategy regime_switch --universe-limit 10 --interval 1h --period 6M --report-dir reports
```

Notes:
- `pre-commit` hooks are configured in `trading-bot/.pre-commit-config.yaml`, but `pre-commit` itself is not pinned in `requirements-dev.txt`.
- Reports are written under `trading-bot/reports/` by default.

## Architecture

### Data Fetching & Normalization

- Entry point: `trading_bot.data_fetcher.fetch_data_online()` (Yahoo/Binance).
- Resilience: circuit breakers + bounded retries/backoff; optional caching under `trading-bot/cache/`.
- Output invariants: tz-aware UTC `DatetimeIndex`, sorted ascending, de-duplicated, OHLCV coerced to numeric, NaN rows dropped.

### Strategy System

All strategies inherit from `StrategyBase` (`trading-bot/src/trading_bot/strategies/strategy_base.py`):
- `execute()` → orchestrates data prep/validation/indicators/strategy run
- `_prepare_data()` → copy & sort data
- `_validate_data()` → min rows check
- `_add_indicators()` → hook for indicator calculation
- `_run_strategy()` → abstract method implemented by each strategy

Multi-strategy mode calls `generate_signal()` → `_compute_signal_action()` to produce `SignalDecision` votes.

**Available strategies:**
- Technical: `atr_breakout`, `mean_reversion`, `regime_switch`
- ML (tabular): `day_trading_ml` (XGBoost/HistGB), `short_term` (XGBoost), `mid_term`/`long_term` (RandomForest)
- ML (deep learning): `day_trading` (LSTM)

Strategies are lazily imported via the registry in `trading-bot/src/trading_bot/strategy_manager.py` to avoid importing heavy ML dependencies until needed.

### Key Patterns

**Inference-only enforcement:** ML strategies automatically set `inference_only=True` during runtime (`trading-bot/src/trading_bot/main.py`) to prevent accidental retraining. Training is done via `trading-bot/src/trading_bot/train.py`.

**LSTM benchmark gating:** `day_trading` (LSTM) skips trades if `day_trading_ml` (tabular) performs better; controlled by benchmark metrics in saved model metadata.

**Multi-strategy aggregation:** In multi-strategy mode (`--strategies`), each strategy produces a `SignalDecision` vote; `SignalAggregator` combines votes deterministically (RISK_EXIT > EXIT > long_term filter > mid_term bias > majority vote).

**Paper trading isolation:** Each strategy maintains separate state files with file locking (`paper_trading_state_<strategy>.json`) to prevent multi-process conflicts.

**Model persistence keying:** Model artifacts are scoped by `build_persistence_key(strategy, data_source, ticker, interval)` so different markets/timeframes do not share artifacts.

**Config drift detection:** Warns when runtime config hash differs from the hash stored with a trained model; `force_retrain_on_drift` can purge stale artifacts.

**Circuit breaker:** Data fetching uses pybreaker to handle API failures gracefully.

### Source Layout

```
trading-bot/src/trading_bot/
├── main.py              # Paper trading entry point
├── train.py             # ML training entry point
├── config_handler.py    # Config loading & sleep duration
├── data_fetcher.py      # Yahoo/Binance data with circuit breaker
├── strategy_manager.py  # Lazy strategy instantiation
├── core/                # Custom exceptions
├── interfaces/          # Signal source abstractions (multi-strategy)
├── strategies/          # Strategy implementations
├── indicators/          # RSI, MACD, SMA, EMA, BB, ADX, Stochastic
├── backtest/            # Engine, metrics, portfolio tools
├── models/              # Pydantic configs, LSTM builder
├── utils/               # Paper trading, model persistence, validators
└── configs/             # Strategy config files (config_<strategy>.py)
```

### Configuration

Strategy configs in `trading-bot/src/trading_bot/configs/config_<strategy>.py` define a `CONFIG` dict and are validated by `trading_bot.models.config.StrategyConfig`.

Environment variables (`trading-bot/.env`, see `trading-bot/.env.example`): Binance API keys, SMTP settings for notifications, cache TTL, log level.

Useful runtime env overrides (optional):
- `TRADING_BOT_SEED` (reproducibility)
- `TRADING_BOT_STATE_DIR` (paper trading state location)
- `TRADING_BOT_MIN_ROWS`, `TRADING_BOT_SLEEP_SECONDS`, `TRADING_BOT_MIN_SLEEP_SECONDS` (runtime defaults)

Pydantic models in `models/config.py` and `models/env_settings.py` handle validation.

### Artifacts & State

- Data cache: `trading-bot/cache/` (gzipped pickle frames) and universe cache JSON.
- Model artifacts: `trading-bot/saved_models/` (versioned artifacts under `<persistence_key>/`; file-locked).
- Reports: `trading-bot/reports/` (backtests + training reports).
- Paper trading state: `paper_trading_state_<strategy>.json` (single) or `paper_trading_state.json` (multi); file-locked and stored under `TRADING_BOT_STATE_DIR` if set.

## Code Style

- Line length: 120 chars (`black` + `ruff`)
- Lint: `ruff check .`
- Format: `ruff format .` and `black .` (keep diffs focused; avoid drive-by reformatting)
- Strict mypy on: `trading_bot.config_handler`, `trading_bot.models.config`, `trading_bot.models.env_settings`, `trading_bot.utils.logger`, `trading_bot.utils.validators`, `trading_bot.utils.time_utils`, `trading_bot.strategy_manager`
- Pre-commit hooks configured (see `trading-bot/.pre-commit-config.yaml`): black, ruff, ruff-format, mypy

## Professional Development Instructions (Senior Quality Bar)

### How to Work in This Repo

- Prefer the smallest safe change; keep diffs focused and avoid drive-by refactors.
- Before coding, confirm intent and constraints: strategy name, asset/universe, timeframe, data source, and whether the change targets paper trading, backtesting, or training.
- Follow existing patterns (StrategyBase hooks, lazy loading registry, Pydantic config validation) unless there’s a clear problem they can’t solve.
- Avoid introducing new dependencies unless necessary; if needed, explain tradeoffs and impact on packaging/runtime.

### Correctness & Trading-Specific Pitfalls

- Prevent look-ahead bias: indicators, labels, and signals must be computed using only information available at that timestamp.
- Keep time handling unambiguous: use timezone-aware UTC datetimes, sort data ascending, and be explicit about candle alignment/boundaries.
- Preserve separation of concerns: training (`trading-bot/src/trading_bot/train.py`) must not happen during inference/runtime (`trading-bot/src/trading_bot/main.py`).
- Don’t leak data across splits: fit scalers/encoders on train only; prefer time-series CV (see `train_or_load_pipeline()` in `trading-bot/src/trading_bot/utils/strategy_helpers.py`).
- When changing backtests, keep net-of-costs logic intact (fees + slippage) and ensure no-lookahead windows.

### Reliability & Safety

- Validate early and loudly for config/data issues (missing columns, insufficient rows, NaNs after indicator calc); fail fast with clear error messages.
- External calls must be resilient: timeouts, bounded retries, and respect rate limits; avoid infinite loops and silent fallbacks.
- Never log secrets (API keys, emails, tokens); don’t add `.env` contents to code or docs.

### Tests, Quality Gates, and Deliverables

- For any behavior change, add/adjust tests (unit tests for indicators/validators; regression-style tests for backtest metrics when feasible).
- Keep tests offline and deterministic: prefer synthetic data + monkeypatch over network calls.
- Run formatting and static checks on touched code (`ruff`, `black`, `mypy` where applicable) and keep the project’s style consistent.
- Update docs/config examples when user-facing CLI flags, configs, or strategy behavior changes.

### Version Control (Commits & PRs)

- Commit small, coherent units (one logical change per commit); avoid mixing formatting-only changes with behavior changes.
- Commit at natural milestones: after a refactor, after a bug fix + regression test, after wiring a new strategy/config, and before switching context.
- Prefer green commits: run `/tb.check` (or at least targeted tests) before committing; keep `main`/default branch releasable.
- Use consistent messages (Conventional Commits recommended): `feat: ...`, `fix: ...`, `refactor: ...`, `test: ...`, `docs: ...`.
- For trading-logic or data changes, include “why” + how to reproduce in the commit body (strategy/ticker/interval + backtest/train command).
- Never commit secrets or runtime artifacts (`.env`, `cache/`, `saved_models/`, `reports/`, `paper_trading_state*.json`).

### Adding a New Strategy (Checklist)

- Add `trading-bot/src/trading_bot/configs/config_<strategy>.py` with `CONFIG`.
- Register in `trading-bot/src/trading_bot/strategy_manager.py` (lazy import registry).
- Add to `trading-bot/src/trading_bot/models/config.py` (`ALLOWED_STRATEGIES`) and ensure Pydantic validation still passes.
- Add tests for selection/execution and (if used in multi-strategy mode) `generate_signal()`/aggregation.

### Adding a New Indicator (Checklist)

- Implement the indicator in `trading-bot/src/trading_bot/indicators/` and add basic unit tests.
- Register the name in `trading-bot/src/trading_bot/models/config.py` (`ALLOWED_INDICATORS`) so configs validate correctly.

### Review Checklist (Before You Say “Done”)

- Does this change alter trading logic? If yes, confirm no look-ahead bias and no leakage.
- Are edge cases covered (empty data, partial candles, API failures, NaNs, bad configs)?
- Are logs actionable (context-rich, not noisy) and errors explicit?
- Is the change compatible with both backtest and paper trading where relevant?
