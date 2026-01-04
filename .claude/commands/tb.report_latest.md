---
description: Summarize the latest JSON report under `trading-bot/reports/` (train/backtest/portfolio).
---

## User Input

```text
$ARGUMENTS
```

## Flow

1. Work from `trading-bot/`.
2. Choose a report type filter from `$ARGUMENTS` (optional):
   - `train` → `train_*.json`
   - `backtest` → `backtest_*.json`
   - default: any `*.json` in `reports/`
3. Find the newest matching report by mtime.
4. Print:
   - Path
   - High-level params (strategy/ticker/interval/period/source)
   - Key metrics (handles missing keys gracefully)

## Suggested implementation (terminal)

```bash
cd trading-bot
python3 - <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

reports = Path("reports")
if not reports.exists():
    raise SystemExit("No reports/ directory found.")

mode = (sys.argv[1] if len(sys.argv) > 1 else "").lower()

files = sorted(reports.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if mode == "train":
    files = [p for p in files if p.name.startswith("train_")]
elif mode == "backtest":
    files = [p for p in files if p.name.startswith("backtest_")]

if not files:
    raise SystemExit("No matching report JSON files found.")

path = files[0]
payload = json.loads(path.read_text(encoding="utf-8"))

def pick(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}

print("report:", path)
print("strategy:", pick(payload, "strategy"))
print("ticker:", pick(payload, "ticker"))
print("interval:", pick(payload, "interval"))
print("period:", pick(payload, "period"))
print("data_source:", pick(payload, "data_source"))
print("bars:", pick(payload, "bars"))
print("trades:", pick(payload, "trades"))

if metrics:
    keys = [
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "sortino",
        "hit_rate",
        "avg_trade_pnl",
        "avg_trade_return",
        "turnover",
    ]
    print("metrics:")
    for k in keys:
        if k in metrics:
            print(f"  - {k}: {metrics.get(k)}")
else:
    # training report layout often nests metrics
    m = payload.get("metrics", {})
    if isinstance(m, dict) and m:
        print("metrics:")
        for k, v in m.items():
            print(f"  - {k}: {v}")

if "config_signature" in payload:
    print("config_signature:", payload.get("config_signature"))
if "dataset_hash" in payload:
    print("dataset_hash:", payload.get("dataset_hash"))
PY
```

Tip: run `/tb.report_latest train` or `/tb.report_latest backtest`.

