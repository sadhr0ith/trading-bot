---
description: Start a local Prometheus `/metrics` endpoint for this repo (long-running; requires explicit --run).
---

## User Input

```text
$ARGUMENTS
```

## Safety

- This starts a long-running process. Do not run unless `$ARGUMENTS` contains `--run` (or `run`).

## Inputs

- Optional: `--port 8000` (default 8000)
- Optional: `--addr 127.0.0.1` (default bind all interfaces)

## Flow

1. Work from `trading-bot/`.
2. Build the command that starts the server and keeps the process alive:
   - `TradingMetrics.start_http_server(...)` starts the server
   - Keep the process alive with `time.sleep(...)`
3. If `--run` is present, start it. Otherwise print the command to run.
4. After start, print how to verify:
   - `curl http://127.0.0.1:<port>/metrics`

## Suggested implementation (terminal)

```bash
cd trading-bot
python3 - <<'PY'
from __future__ import annotations

import argparse
import time

from trading_bot.utils.metrics import TradingMetrics

parser = argparse.ArgumentParser()
parser.add_argument("--run", action="store_true")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--addr", default="")
args, _ = parser.parse_known_args()

if not args.run:
    raise SystemExit("Dry-run only. Re-run with --run to start the metrics server.")

TradingMetrics.start_http_server(port=args.port, addr=args.addr)
print(f"Serving /metrics on http://{args.addr or '127.0.0.1'}:{args.port}/metrics (Ctrl-C to stop)")
while True:
    time.sleep(3600)
PY
```
