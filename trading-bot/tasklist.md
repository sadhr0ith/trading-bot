# Trading Bot (Crypto) — Detailed Implementation Tasklist

This file is a concrete, opinionated roadmap for future development.
It intentionally makes decisions (no open questions) so work can start immediately.

## Chosen Defaults (for future-proof development)

- **Market**: crypto
- **Execution mode**: paper trading only (no real orders)
- **Instruments (start)**: `BTCUSDT`, `ETHUSDT`
- **Universe (later)**: top 10 spot pairs by 30d volume (USDT-quoted, excludes stable-stable)
- **Positioning (start)**: spot, long-only; no shorts/leverage until backtests + risk are solid
- **Primary timeframes**:
  - Intraday: `15m` (mean reversion)
  - Core: `1h` (trend + ML)
  - Regime confirmation: `4h`
  - Allocation/filters: `1d`
- **Cost model (defaults)**:
  - Fee: `0.10%` per side (taker-style)
  - Slippage: `0.02%` per side (simple linear)
  - Spread: ignored initially (captured by slippage)

## Global Definition of Done

A strategy/change is “done” when:
- It runs end-to-end with deterministic outputs on cached OHLCV.
- There is a minimal test (unit or integration) preventing obvious leakage/regressions.
- Backtest report includes net-of-costs metrics.
- Config is validated by `StrategyConfig` and is documented in `README.md` or `quick-start.md`.

---

## Milestone 1 — Backtest Foundation (must-have)

Goal: evaluate strategies on crypto OHLCV **net of costs** with correct time-series hygiene.

### 1.1 Create a dedicated backtest package
- [ ] Add `src/trading_bot/backtest/__init__.py`
- [ ] Add `src/trading_bot/backtest/engine.py`
  - Event loop over bars.
  - Input: OHLCV `pd.DataFrame` (DatetimeIndex UTC, sorted).
  - Output: trades list + equity curve + per-bar positions.
- [ ] Add `src/trading_bot/backtest/costs.py`
  - Fee + slippage functions (per-side).
  - Apply costs on fills.
- [ ] Add `src/trading_bot/backtest/metrics.py`
  - Compute: total return, CAGR (if applicable), max drawdown, Sharpe/Sortino, hit rate, avg trade, turnover.
- [ ] Add `src/trading_bot/backtest/report.py`
  - Write a JSON report and a compact console summary.

Acceptance:
- Run on a cached frame and produce a report file under `reports/`.

### 1.2 Define a clean strategy interface for backtests (no leakage)
- [ ] Add `src/trading_bot/backtest/interfaces.py` with a minimal API:
  - `on_bar(state, window_df) -> Signal` where `window_df` is past-only.
  - Signal supports: `BUY`, `SELL`, `HOLD` (long-only initially).
- [ ] Add `src/trading_bot/backtest/adapters.py`
  - Adapter to reuse existing indicator functions.
  - Do **not** call existing `StrategyBase.execute()` per bar (too heavy, too stateful).

Acceptance:
- A dummy strategy can trade and backtest deterministically.

### 1.3 Add a backtest CLI
- [ ] Add `src/trading_bot/backtest/cli.py` (module runnable)
  - Example usage:
    - `python -m trading_bot.backtest.cli --strategy atr_breakout --ticker BTCUSDT --interval 1h --period 1y`
- [ ] Ensure it uses `fetch_data_online()` but with `cache_ttl_seconds=0` when backtesting to avoid serving stale data inadvertently.

Acceptance:
- CLI runs offline if cache exists; otherwise prints a clear message.

### 1.4 Tests for backtest correctness
- [ ] Add `tests/test_backtest_engine.py`
  - Deterministic fills, cost application, and PnL math sanity checks.
- [ ] Add `tests/test_no_lookahead.py`
  - Guards that strategy windows never include future rows.

---

## Milestone 2 — Add Strong Non-ML Baselines (production-grade)

Goal: have robust, interpretable baselines that ML must beat net-of-costs.

### 2.1 ATR Breakout Trend Strategy (1h/4h)
- [ ] Add `src/trading_bot/strategies/atr_breakout_strategy.py`
  - Entry: Donchian breakout (N bars) OR close > rolling high.
  - Stop: ATR-based initial stop + trailing stop.
  - Exit: trailing stop OR time stop.
- [ ] Add config `src/trading_bot/configs/config_atr_breakout.py`
- [ ] Wire into `src/trading_bot/strategy_manager.py` registry.
- [ ] Add backtest adapter implementation.

Acceptance:
- Backtest runs and produces trades; no lookahead; stable results.

### 2.2 Mean Reversion Strategy (15m/1h)
- [ ] Add `src/trading_bot/strategies/mean_reversion_strategy.py`
  - Entry: BB lower-band breach + RSI oversold confirmation.
  - Exit: mean reversion to BB mid + time stop; optional partial TP.
  - Strict max loss per trade.
- [ ] Add config `src/trading_bot/configs/config_mean_reversion.py`
- [ ] Wire into registry + backtest.

### 2.3 Regime Switcher (ADX/vol filter)
- [ ] Add `src/trading_bot/strategies/regime_switch_strategy.py`
  - Compute regime: trend vs range vs high-vol no-trade.
  - Route to ATR breakout or mean reversion.
- [ ] Config includes thresholds and cooldowns.

Acceptance:
- Backtest shows reduced drawdowns vs always-on.

---

## Milestone 3 — Fix Day-Trading ML (align model to trading)

Goal: make ML meaningful by predicting **returns/direction**, not next close.

### 3.1 Define targets that match decisions
- [ ] For intraday ML, define label as forward return:
  - `ret_h = Close.pct_change(h).shift(-h)` with `h ∈ {1, 3}` bars.
  - Decision is based on predicted return net of costs.

Acceptance:
- A naive baseline (predict 0 return / last return) is implemented and reported.

### 3.2 Build a tabular baseline first (XGBoost/GBDT)
- [ ] Add `src/trading_bot/strategies/day_trading_ml_strategy.py`
  - Pipeline similar to `short_term_strategy`, but intraday horizon.
  - Features: lag returns, rolling vol, ATR, RSI, BB width, regime features.
  - CV: `TimeSeriesSplit` with gap = horizon.
- [ ] Add config `src/trading_bot/configs/config_day_trading_ml.py`
- [ ] Add model persistence + drift signatures.

Acceptance:
- Beats non-ML baselines on at least one regime slice OR is safely gated to no-trade.

### 3.3 LSTM only as a v2 (optional, gated)
- [ ] Refactor existing LSTM day trading to:
  - Predict forward return/direction.
  - Evaluate using hit-rate-at-threshold + net PnL proxy, not price MAE.
- [ ] Train on multi-asset pool (BTC/ETH + 3–8 liquid alts) to increase sample size.

Acceptance:
- Only enabled if it beats XGBoost baseline net-of-costs.

### 3.4 Risk + gating for ML strategies
- [ ] Extend quality gates:
  - Require ML to beat baseline by margin **and** meet max drawdown constraints.
  - Add minimum expected edge: `abs(pred_ret) > fee+slippage+threshold`.
- [ ] Add “no-trade when uncertain” behavior.

---

## Milestone 4 — Production Patterns (still paper-trading)

Goal: stable operation, reproducibility, observability.

### 4.1 Separate training from inference
- [ ] Add `python -m trading_bot.train --strategy day_trading_ml` command.
- [ ] Ensure live loop runs `inference_only=True` by default for ML strategies.

### 4.2 Experiment tracking (lightweight)
- [ ] Persist per-run metadata: dataset hash, feature version, horizon, costs, metrics.
- [ ] Write reports to `reports/` with timestamp + config signature.

### 4.3 Observability & drift
- [ ] Add a small “data health” report per run:
  - missing bars, gaps, volatility regime.
- [ ] Add drift checks:
  - feature distribution shift
  - performance decay vs baseline.

---

## Milestone 5 — Multi-asset Portfolio (crypto-specific)

Goal: avoid single-asset brittleness and manage risk across a basket.

- [ ] Add universe loader (top N by volume, cached).
- [ ] Add portfolio constraints:
  - max positions
  - per-asset max exposure
  - correlation/vol targeting (simple first)
- [ ] Add portfolio-level backtest metrics.

---

## Implementation Order (strict)

1) Milestone 1 (backtest foundation) — without it, ML changes are blind.
2) Milestone 2 (non-ML baselines) — provides a benchmark and fallback.
3) Milestone 3 (day-trading ML refactor) — only after baselines exist.
4) Milestone 4/5 as hardening and expansion.

## Git Workflow (recommended)

- One PR/branch per milestone.
- Commit granularity: one coherent unit (new module + tests) per commit.
- Always include:
  - a runnable command example
  - a minimal test
  - updated config + docs.
