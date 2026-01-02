# Trading Bot

Multi-strategy trading bot (crypto/stocks) with ML pipelines, technical indicators, paper trading, and backtesting.

## Requirements
- Python 3.12 (recommended)
- Dependencies from `requirements.txt` (prod) and `requirements-dev.txt` (dev/test)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# for development
pip install -r requirements-dev.txt
```

## Configuration
1) Copy `.env.example` -> `.env` and fill in:
   - `BINANCE_API_KEY` / `BINANCE_API_SECRET` (optional for Binance)
   - `GMAIL_SENDER_EMAIL` / `GMAIL_APP_PASSWORD` (email notifications)
   - `NOTIFICATION_EMAILS` (comma-separated list)
   - `TRADING_BOT_CACHE_TTL_SECONDS` (optional fetch cache)
2) Choose/edit a strategy config in `configs/config_<strategy>.py`.

## Run
```bash
python main.py --strategy short_term
python main.py --strategy day_trading_ml
# available: day_trading | day_trading_ml | short_term | mid_term | long_term | atr_breakout | mean_reversion | regime_switch
```

## Backtest
```bash
python -m trading_bot.backtest.cli --strategy atr_breakout --ticker BTCUSDT --interval 1h --period 1y
```

## Portfolio Backtest (CLI)
```bash
python -m trading_bot.backtest.portfolio_cli --strategy atr_breakout --tickers BTCUSDT,ETHUSDT --interval 1h --period 1y
python -m trading_bot.backtest.portfolio_cli --strategy regime_switch --universe-limit 10 --interval 1h --period 6M
```

## ML Training (separate from inference)
```bash
python -m trading_bot.train --strategy day_trading_ml
python -m trading_bot.train --strategy day_trading
```
- The live loop enforces `inference_only=True` for ML strategies; training is run via the command above.
- `day_trading` (LSTM) is benchmark-gated against `day_trading_ml` by default; if the benchmark model/metrics are missing or better, LSTM trades are skipped.

## Portfolio Backtest (multi-asset)
```python
from trading_bot.backtest.portfolio import PortfolioConstraints, run_portfolio_backtest

constraints = PortfolioConstraints(
    max_positions=5,
    max_exposure_per_asset=0.2,
    vol_targeting=True,
    correlation_filter=True,
    max_pairwise_correlation=0.9,
)
result = run_portfolio_backtest(data_by_asset, strategy_factory, constraints=constraints)
print(result.metrics)
```

## Features
- Strategies:
  - `day_trading` (LSTM, intraday returns)
  - `day_trading_ml` (tabular ML, intraday returns)
  - `short_term` (XGBoost, 5-day horizon)
  - `mid_term` (RandomForest, 20-day horizon)
  - `long_term` (RandomForest, 50-day horizon)
  - `atr_breakout` (Donchian + ATR)
  - `mean_reversion` (BB + RSI)
  - `regime_switch` (trend/range + vol filter)
- Indicators: RSI, MACD, SMA/EMA, Bollinger Bands, ADX, Stochastic.
- Paper trading with isolated state per strategy, SL/TP/fee handling.
- Model persistence (sklearn/keras) and config drift detection.
- Data validation (Pydantic + OHLCV sanity checks).
- Backtest with net-of-costs metrics and JSON reports.
- Portfolio backtest (multi-asset, exposure constraints).

## Structure
```
trading-bot/
├── configs/           # strategy configs
├── strategies/        # strategy implementations + StrategyBase
├── indicators/        # technical indicators
├── utils/             # validators, logging, cache, persistence, paper trading
├── models/            # Pydantic models + LSTM builder
├── backtest/          # backtest engine + metrics + report
├── tests/             # unit/integration tests
└── saved_models/      # model artifacts (local, gitignored)
```

## Operational notes
- Paper trading only by default; no real execution.
- Binance data is paginated; cache TTL helps reduce API load.
- Retraining can be expensive (especially LSTM); consider separating training/inference in production.
