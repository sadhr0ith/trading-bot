---
description: Run the no-lookahead/leakage tests and scan for common future-leak patterns (read-only).
---

## User Input

```text
$ARGUMENTS
```

## Goal

Quick confidence check that changes didn’t introduce look-ahead bias or obvious leakage patterns.

## Flow

1. Work from `trading-bot/`.
2. Run targeted tests:
   - `pytest -k lookahead`
3. Optionally run a broader “strategy sanity” subset if `$ARGUMENTS` contains `more`:
   - `pytest -k "strategy and not optional"`
4. Run a lightweight code scan (signal only; not definitive):
   - Search for suspicious shifts and forward-looking operations: `shift(-1)` etc.
5. Report:
   - Test results
   - Any suspicious matches with file/line pointers and a short explanation of whether they’re safe or need review

## Suggested commands

```bash
cd trading-bot
pytest -k lookahead
rg -n "shift\\(-1\\)|\\.shift\\(-1\\)" src/trading_bot
```

