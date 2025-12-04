# Merged After-Refactor Overview

## What remains broken (must-fix)
- Paper trading cash accounting is wrong: BUY subtracts only fee, not notional; SELL adds only PnL without returning principal (`trading-bot/utils/paper_trading.py`). Balances/PnL are unusable.
- Period parsing/helper gap: `tests/test_parse_period_extended.py` imports missing `utils.time_utils`; `trading-bot/data_fetcher.py` parser treats `m` as minutes, lacks `w`, so common inputs fail or mis-map.
- Binance data index: fetched frames keep a numeric index instead of candle time, so calendar and lag features are built on integers (`trading-bot/data_fetcher.py`).
- Short-term inference uses a single-row frame while the pipeline builds lags/rolling features; all such features become NaN→0, so live predictions diverge from training (`trading-bot/strategies/short_term_strategy.py`).
- Hardcoded fallback sender email is a real address; should fail fast if env vars are missing (`trading-bot/utils/email_notifications.py`).
- Risk config keys are easy to misconfigure: code reads `stop_loss`/`take_profit`/`max_position_size`, but configs/tests also use `*_pct`; ignored keys mean stops may be disabled and sizing can explode (`trading-bot/utils/risk_management.py`, `trading-bot/tests/test_risk_paper_integration.py`).

## Secondary issues
- Shared paper trading state path means positions from one strategy can leak into another run (`trading-bot/utils/paper_trading.py` default `paper_trading_state.json`).
- Duplicate/low-signal features in mid/long-term (return and momentum duplicates across the same lags) inflate feature space without adding signal (`trading-bot/strategies/mid_term_strategy.py`, `trading-bot/strategies/long_term_strategy.py`).
- Unused/dead code: model wrappers (`trading-bot/models/*.py`), `backtesting.py`, and config `models` entries are not wired into strategies; consider pruning or integrating.
- Logging consistency: `data_fetcher.py` uses raw `logging.getLogger` while the rest uses `utils.logger.setup_logger`, yielding mixed formatting/levels.

## Notes on rejected findings
- Alleged “data leakage” from computing indicators/lags with the inference row present is a false alarm: `shift`/rolling use past values; the last row does not leak future data into training.
- Backtesting equity/index length mismatch flagged elsewhere is not reproducible; the loop keeps lengths aligned.

## Suggested next steps
1) Fix paper trading cash flow (subtract notional+fee on BUY, return principal−fee on SELL) and add a state path per strategy/profile.  
2) Add `utils/time_utils.parse_period_to_timedelta` with clear units (`m`=month or minute?), support `w`, and align tests/configs.  
3) Set Binance data index to open time before FE; ensure downstream uses DateTimeIndex.  
4) Rework short-term inference to build features on the full recent window (or reuse persisted pipeline with cached history) before predicting.  
5) Remove hardcoded email fallback; fail fast when credentials are missing.  
6) Normalize risk config keys (`*_pct` vs plain fractions) and warn on unknown keys.***
