---
description: Add a new strategy end-to-end (config, implementation, registry, validation, tests) following this repo's architecture.
---

## User Input

```text
$ARGUMENTS
```

## Clarify first (if missing from $ARGUMENTS)

Ask for:
- Strategy name (slug, e.g. `momentum_v2`)
- Type: technical (rules/indicators) vs ML (tabular vs deep)
- Target interval(s) and data source(s) (Yahoo vs Binance)
- Whether it must be backtestable via the CLI (technical strategies only by default)
- Expected signal shape (BUY/SELL/HOLD vs explicit EXIT/RISK_EXIT)

## Implementation checklist (must follow)

1. Add config:
   - Create `trading-bot/src/trading_bot/configs/config_<strategy>.py` with a `CONFIG` dict.
2. Add strategy class:
   - Create `trading-bot/src/trading_bot/strategies/<strategy>_strategy.py`
   - Inherit `StrategyBase`
   - Implement `_run_strategy()` (paper trading loop execution)
   - Implement `_compute_signal_action()` if it should participate in multi-strategy aggregation.
3. Register for lazy loading:
   - Add to `trading-bot/src/trading_bot/strategy_manager.py` registry.
4. Wire into config validation:
   - Add to `trading-bot/src/trading_bot/models/config.py` (`ALLOWED_STRATEGIES`)
5. If backtestable:
   - Implement `on_bar(state, window_df)` and ensure no look-ahead
   - Add to `trading-bot/src/trading_bot/backtest/cli.py` and/or `portfolio_cli.py` registries (if intended)
6. Add tests:
   - `tests/test_strategy_manager.py` (registry selection)
   - A strategy-specific test covering no-lookahead and core decision logic
   - If multi-strategy: add/extend an aggregation test to ensure deterministic behavior
7. Update docs if user-facing:
   - If new CLI-usable strategy, update `trading-bot/README.md` (optional)

## Non-negotiables

- No look-ahead bias; all computations must use data up to the bar timestamp.
- No training during runtime inference; training only via `trading_bot.train`.
- Keep changes minimal and consistent with existing patterns.
