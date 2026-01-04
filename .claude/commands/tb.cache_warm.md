---
description: Warm the market-data cache for one or more tickers/strategies (network + disk writes; requires explicit --run).
---

## User Input

```text
$ARGUMENTS
```

## Safety

- This command fetches data over the network and writes cache files under `trading-bot/cache/`.
- Do not execute unless `$ARGUMENTS` contains `--run` (or `run`).

## Inputs

Supported patterns in `$ARGUMENTS`:

- Strategy-based:
  - `--strategy <name>` (warms the ticker/interval/period/source from that strategy’s config)
- Ticker-based:
  - `--source yahoo|binance`
  - `--tickers BTCUSDT,ETHUSDT` (comma-separated)
  - `--interval 1h`
  - `--period 1y`
  - Optional: `--ttl 3600` (cache TTL seconds; default 3600)

## Flow

1. Work from `trading-bot/`.
2. Resolve what to warm (strategy config or explicit tickers).
3. Build a Python runner that calls `fetch_data_online(...)` for each target and prints:
   - rows, start/end timestamps
4. If `--run` is present: execute it. Otherwise: print the exact command to run.

## Suggested implementation (terminal)

```bash
cd trading-bot
python3 - <<'PY'
from __future__ import annotations

import sys
from typing import Any

from trading_bot.config_handler import load_config
from trading_bot.data_fetcher import fetch_data_online

def has_flag(flag: str) -> bool:
    return flag in sys.argv

def arg_value(flag: str) -> str | None:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None

run = has_flag("--run") or has_flag("run")
strategy = arg_value("--strategy")
source = arg_value("--source")
tickers_raw = arg_value("--tickers")
interval = arg_value("--interval")
period = arg_value("--period")
ttl_raw = arg_value("--ttl") or "3600"

targets: list[dict[str, Any]] = []

if strategy:
    cfg = load_config(strategy)
    if cfg is None:
        raise SystemExit(f"Config not found for strategy '{strategy}'")
    cfgd = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
    targets.append(
        {
            "source": cfgd.get("data_source"),
            "ticker": cfgd.get("ticker"),
            "interval": cfgd.get("interval"),
            "period": cfgd.get("period"),
        }
    )
else:
    if not (source and tickers_raw and interval and period):
        raise SystemExit("Provide either --strategy <name> OR (--source, --tickers, --interval, --period).")
    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
    for t in tickers:
        targets.append({"source": source, "ticker": t, "interval": interval, "period": period})

ttl = int(ttl_raw)
print(f"Targets: {targets}")
print(f"TTL: {ttl}s")
if not run:
    raise SystemExit("Dry-run only. Re-run with --run to fetch and write cache.")

print(f"Warming {len(targets)} target(s)...")
for t in targets:
    df = fetch_data_online(
        source=t["source"],
        ticker=t["ticker"],
        period=t["period"],
        interval=t["interval"],
        cache_ttl_seconds=ttl,
    )
    rows = 0 if df is None else int(len(df))
    start = None if df is None or df.empty else str(df.index.min())
    end = None if df is None or df.empty else str(df.index.max())
    print(f"- {t['source']} {t['ticker']} {t['period']}/{t['interval']}: rows={rows}, start={start}, end={end}")
PY
```

