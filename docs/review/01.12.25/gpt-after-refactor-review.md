# Code Review – trading-bot

## Follow-up check after reported fixes
- Critical cash accounting bug in `utils/paper_trading.py` remains: BUY still only subtracts fee, SELL adds PnL without returning principal; balances stay inflated.
- Period parsing/helper still missing (`utils.time_utils` not present); parser in `data_fetcher.py` still treats `m` as minutes and lacks `w`, so tests in `tests/test_parse_period_extended.py` would still fail.
- Binance data index is unchanged (no DateTimeIndex on open time), so calendar/lags are still computed on a numeric RangeIndex.
- Short-term inference still builds features from a single row, zeroing lagged/rolling features (`strategies/short_term_strategy.py`), so live predictions diverge from training pipeline.
- Risk config key mismatch persists (`stop_loss_pct`/`take_profit_pct` ignored by `RiskManager`); no warnings added.

## Architecture snapshot
- Main loop in `main.py` loads strategy config, validates via `utils.validators`, fetches data (`data_fetcher`) and delegates to a strategy selected in `strategy_manager.py`.
- Strategies share `StrategyBase` for logging, seeding, risk management, and paper execution; persistence handled through `utils.model_persistence` + `utils.strategy_helpers.train_or_load_pipeline`.
- Feature engineering is split between on-the-fly sklearn transformers (short-term) and manual pandas feature building (mid/long-term); paper trading state is stored locally in JSON.

## Major findings
- [Critical] Paper trading never deducts trade notional and ignores fees on exit, so balances grow unrealistically: only the fee is subtracted on BUY and principals are never withdrawn (`utils/paper_trading.py:81-98`), while SELL adds PnL without subtracting the original stake (`utils/paper_trading.py:40-48`). Risk sizing and PnL are therefore materially wrong.
- [Major] Binance fetches return a numeric index; time features then derive bogus calendar values (e.g., 1970 epoch conversions) and risk lags that need history. The index should be the candle open time before feeding pipelines (`data_fetcher.py:63-113`).
- [Major] Short-term inference feeds a single-row frame into a pipeline that builds lags/rolling stats (`strategies/short_term_strategy.py:50-52,191-204`). All lagged features become NaN→0, so live predictions drop historical context and differ from training-time feature generation.
- [Major] Period parsing is inconsistent and the expected helper is missing. Tests import `utils.time_utils.parse_period_to_timedelta`, but that module does not exist (`tests/test_parse_period_extended.py:1-11`). The existing parser only supports `d/h/m/mo/y` and treats `m` as minutes (`data_fetcher.py:13-31`), so common inputs like `1w` or monthly `1m` misbehave/fail.
- [Major] Risk knobs are easy to misconfigure: `RiskManager` looks for `stop_loss`/`take_profit` fractions, but configs/tests also use `stop_loss_pct`/`take_profit_pct` and huge `max_position_size` values (`utils/risk_management.py:18-32`, `tests/test_risk_paper_integration.py:6-13`). These keys are ignored, so stops may be disabled and sizes explode without warning.

## Additional observations
- Momentum and return features are duplicated (same computation) across lags in the mid-term strategy, inflating feature space without new signal (`strategies/mid_term_strategy.py:44-58`). Long-term mirrors this pattern.
- Persistence/inference consistency: strategies load persisted artifacts but reuse a shared `paper_trading_state.json` for all strategies, so positions from one strategy leak into another run (default path in `utils/paper_trading.py:15-21`).
- Unused/dead code: model wrappers in `models/*.py`, `backtesting.py`, and config `models` entries are not referenced by any strategy; `NotFittedError` imports go unused in mid/long-term strategies. Consider pruning or wiring them up.
- Logging/emails are uneven: mid-term strategy omits email notifications while others send alerts; data fetching uses raw `logging.getLogger` instead of the configured `setup_logger`, leading to mixed formatting/levels.
