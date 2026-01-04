---
description: Purge persisted artifacts for a given persistence key (DESTRUCTIVE; requires explicit --yes).
---

## User Input

```text
$ARGUMENTS
```

## Safety (non-negotiable)

- This command deletes files under `trading-bot/saved_models/` and cannot be undone.
- Do not purge unless `$ARGUMENTS` contains `--yes` (or `yes`) AND a target key.

## Inputs

- `--key <persistence_key>` (recommended)
- Or provide the key as a bare token in `$ARGUMENTS`

## Flow

1. Work from `trading-bot/`.
2. Resolve the target key.
3. Print what will be deleted:
   - `saved_models/<key>/` directory (if present)
   - Any legacy single-file artifacts (handled by `ModelPersistence.purge`)
4. Require explicit confirmation:
   - If `--yes` is not present, stop after printing the plan.
5. Execute purge via `ModelPersistence().purge(key)`.
6. Verify the directory is gone (or empty) and report.

## Suggested implementation (terminal)

```bash
cd trading-bot
python3 - <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from trading_bot.utils.model_persistence import ModelPersistence

argv = sys.argv[1:]
yes = ("--yes" in argv) or ("yes" in argv)

def pick_key(args: list[str]) -> str | None:
    if "--key" in args:
        idx = args.index("--key")
        if idx + 1 < len(args):
            return args[idx + 1]
    for token in args:
        if token.startswith("-"):
            continue
        return token
    return None

key = pick_key(argv)
if not key:
    raise SystemExit("ERROR: provide --key <persistence_key> (or a bare key).")

target_dir = Path("saved_models") / key
print("Target key:", key)
print("Would purge:", target_dir)
if target_dir.exists():
    files = sorted(target_dir.rglob("*"))
    print("Files to delete (preview):")
    for p in files[:50]:
        print(" -", p)
    if len(files) > 50:
        print(f" - ... ({len(files) - 50} more)")
else:
    print("Note: target directory does not exist; purge may still remove legacy artifacts.")

if not yes:
    raise SystemExit("Dry-run only. Re-run with --yes to actually delete.")

ModelPersistence().purge(key)
print("Purge completed.")
print("Exists after purge:", target_dir.exists())
PY
```

