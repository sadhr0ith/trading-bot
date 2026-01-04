---
description: Diagnose and fix a failing test or runtime error with a minimal reproduction, then add/adjust tests to prevent regressions.
---

## User Input

```text
$ARGUMENTS
```

## Flow

1. Reproduce:
   - If `$ARGUMENTS` includes a test nodeid or file, run targeted pytest for it.
   - Otherwise, ask for the failing command + full traceback.
2. Isolate:
   - Identify the smallest failing unit (function/module) and the triggering inputs.
   - Prefer synthetic data and deterministic tests.
3. Fix:
   - Make the minimal change that resolves the root cause.
   - Avoid drive-by refactors.
4. Prove:
   - Re-run the targeted test(s) and then `pytest`.
5. Report:
   - Root cause summary
   - Files changed
   - Tests added/updated

## Repo specifics (must follow)

- Be extra careful about look-ahead/leakage in any strategy/data/ML change.
- Do not introduce network dependencies into tests.
