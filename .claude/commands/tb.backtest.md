---
description: Run a single-asset or portfolio backtest via the built-in CLIs and summarize the resulting report/metrics.
---

## User Input

```text
$ARGUMENTS
```

## Notes (repo-specific)

- The backtest CLIs support technical strategies only: `atr_breakout`, `mean_reversion`, `regime_switch`.
- The engine is long-only and net-of-costs (fee + slippage).

## Flow

1. Work from `trading-bot/`.
2. Decide mode from `$ARGUMENTS`:
   - Portfolio if `$ARGUMENTS` contains `--tickers`, `--universe-limit`, or `portfolio`.
   - Otherwise run single-asset backtest.
3. Build a command with sensible defaults if not provided:
   - `--strategy` (required)
   - Single-asset: `--ticker` (default from config), `--interval`, `--period`, `--data-source`
   - Portfolio: `--tickers` or `--universe-limit`, plus `--interval`, `--period`, `--data-source`
   - Always set `--report-dir reports`
4. If `$ARGUMENTS` contains `run` or `--run`, execute the command; otherwise print the command and ask for confirmation.
5. After running, locate the newest report under `reports/` and summarize:
   - Total return, max drawdown, Sharpe/Sortino, hit rate, avg trade pnl
   - Bars, trades, and the params used

## Output requirements

- Print the exact command used (or proposed).
- Print the report path.
- Provide a short metric summary and any red flags (too few bars, suspicious gaps, etc.).
