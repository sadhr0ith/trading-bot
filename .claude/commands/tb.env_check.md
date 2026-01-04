---
description: Check `.env` against `.env.example` (missing keys, empty values) without printing secrets.
---

## User Input

```text
$ARGUMENTS
```

## Safety

- Never print secret values. Only print key names and high-level status.

## Flow

1. Work from `trading-bot/`.
2. If `.env` does not exist:
   - Recommend `cp .env.example .env`
   - List keys required by `.env.example`
3. If `.env` exists:
   - Parse both files (ignore comments/blank lines).
   - Report:
     - Keys present in `.env.example` but missing in `.env`
     - Keys present but empty in `.env`
     - Extra keys in `.env` (allowed, just informational)
4. If `$ARGUMENTS` contains `fix` or `--fix`:
   - Create `.env` from `.env.example` if missing
   - Append missing keys as `KEY=` (empty) without overwriting existing keys

## Suggested implementation (terminal)

Run from repo root:

```bash
cd trading-bot
python3 - <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

def parse_env(path: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        out[key] = value
    return out

root = Path(".")
example = root / ".env.example"
env = root / ".env"

fix = ("--fix" in sys.argv) or ("fix" in sys.argv)

example_keys = parse_env(example)
env_keys = parse_env(env)

missing = sorted(k for k in example_keys if k not in env_keys)
empty = sorted(k for k, v in env_keys.items() if v in ("", None) and k in example_keys)
extra = sorted(k for k in env_keys if k not in example_keys)

if fix:
    if not example.exists():
        raise SystemExit("ERROR: .env.example not found.")

    if not env.exists():
        env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print("Created .env from .env.example (no values printed).")
    elif missing:
        existing = env.read_text(encoding="utf-8")
        needs_newline = bool(existing) and not existing.endswith("\n")
        with env.open("a", encoding="utf-8") as handle:
            if needs_newline:
                handle.write("\n")
            for key in missing:
                handle.write(f"{key}=\n")
        print(f"Appended {len(missing)} missing key(s) to .env (no values printed).")

    # Recompute after any fix
    env_keys = parse_env(env)
    missing = sorted(k for k in example_keys if k not in env_keys)
    empty = sorted(k for k, v in env_keys.items() if v in ("", None) and k in example_keys)
    extra = sorted(k for k in env_keys if k not in example_keys)

print(f".env present: {env.exists()}")
print(f"Example keys: {len(example_keys)}")
print(f"Missing keys: {missing}")
print(f"Empty values (for example keys): {empty}")
print(f"Extra keys (informational): {extra}")
PY
```
