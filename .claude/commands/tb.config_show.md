---
description: Show the effective validated config for a strategy (after runtime defaults) and the derived persistence key.
---

## User Input

```text
$ARGUMENTS
```

## Flow

1. Identify the strategy:
   - Accept `--strategy <name>` or a bare strategy name in `$ARGUMENTS`.
   - If missing, ask which strategy to inspect.
2. Work from `trading-bot/`.
3. Load the effective config using `trading_bot.config_handler.load_config(strategy)`.
4. Print:
   - The full config as JSON (safe; no secrets expected in config)
   - The derived `persistence_key` (`build_persistence_key(strategy, data_source, ticker, interval)`)
   - A short “runtime defaults” summary: `min_rows`, `cache_ttl_seconds`, `sleep_seconds`
5. If config fails validation, print the exact validation errors and stop.

## Suggested implementation (terminal)

```bash
cd trading-bot
python3 - <<'PY'
from __future__ import annotations

import json
import sys
from typing import Any

from trading_bot.config_handler import load_config
from trading_bot.utils.strategy_helpers import build_persistence_key

def pick_strategy(argv: list[str]) -> str | None:
    if "--strategy" in argv:
        idx = argv.index("--strategy")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    for token in argv:
        if token.startswith("-"):
            continue
        return token
    return None

strategy = pick_strategy(sys.argv[1:])
if not strategy:
    print("ERROR: provide a strategy, e.g. `tb.config_show --strategy short_term`", file=sys.stderr)
    raise SystemExit(2)

cfg = load_config(strategy)
if cfg is None:
    print(f"ERROR: failed to load config for strategy '{strategy}'", file=sys.stderr)
    raise SystemExit(1)

cfg_dict: dict[str, Any] = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
key = build_persistence_key(
    strategy=strategy,
    data_source=cfg_dict.get("data_source"),
    ticker=cfg_dict.get("ticker"),
    interval=cfg_dict.get("interval"),
)

print("persistence_key:", key)
print("runtime_defaults:", {k: cfg_dict.get(k) for k in ("min_rows", "cache_ttl_seconds", "sleep_seconds")})
print(json.dumps(cfg_dict, indent=2, sort_keys=True, default=str))
PY
```

