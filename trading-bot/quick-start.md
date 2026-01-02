# Quick Start (Paper Trading + Backtests)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

Optional `.env` keys:
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` (not required for public market data)
- `NOTIFICATION_EMAILS` (comma-separated)

## Run (paper trading loop)
```bash
python -m trading_bot.main --strategy atr_breakout
python -m trading_bot.main --strategy day_trading_ml
python -m trading_bot.main --strategy day_trading
```

## Train ML strategies (separate from inference)
```bash
python -m trading_bot.train --strategy day_trading_ml
python -m trading_bot.train --strategy day_trading
```

## Single-asset backtest
```bash
python -m trading_bot.backtest.cli --strategy atr_breakout --ticker BTCUSDT --interval 1h --period 1y
```

## Portfolio backtest
```bash
python -m trading_bot.backtest.portfolio_cli --strategy regime_switch --universe-limit 10 --interval 1h --period 6M
python -m trading_bot.backtest.portfolio_cli --strategy atr_breakout --tickers BTCUSDT,ETHUSDT --interval 1h --period 1y
```

Reports are written under `reports/`.

