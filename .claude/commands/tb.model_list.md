---
description: List persisted model artifacts under `trading-bot/saved_models/` and summarize latest metadata per key.
---

## User Input

```text
$ARGUMENTS
```

## Flow

1. Work from `trading-bot/`.
2. Optionally accept a substring filter in `$ARGUMENTS` to narrow keys (e.g. `short_term__`).
3. For each key directory under `saved_models/`:
   - Find the newest `*.json` metadata file
   - Print a compact summary: key, version, trained_until, feature_columns count, and common metrics if present

## Suggested implementation (terminal)

```bash
cd trading-bot
python3 - <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

base = Path("saved_models")
if not base.exists():
    raise SystemExit("No saved_models/ directory found.")

needle = " ".join(sys.argv[1:]).strip().lower()

dirs = [p for p in base.iterdir() if p.is_dir() and p.name != ".lock"]
dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

def safe_load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

for d in dirs:
    key = d.name
    if needle and needle not in key.lower():
        continue
    metas = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not metas:
        print(f"- {key}: (no metadata json)")
        continue
    meta_path = metas[0]
    meta = safe_load(meta_path)
    feature_cols = meta.get("feature_columns")
    n_features = len(feature_cols) if isinstance(feature_cols, list) else None
    print(
        f"- {key}: version={meta.get('version', meta_path.stem)} "
        f"trained_until={meta.get('trained_until')} features={n_features} "
        f"cv_mae={meta.get('cv_mae') or meta.get('mae_cv')} "
        f"baseline_mae={meta.get('baseline_mae')} pnl_proxy_net={meta.get('pnl_proxy_net')}"
    )
PY
```

