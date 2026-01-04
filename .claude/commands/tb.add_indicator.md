---
description: Add a new technical indicator module (implementation + tests + config validation) in the existing indicator framework.
---

## User Input

```text
$ARGUMENTS
```

## Flow

1. Clarify the indicator name and output shape:
   - Single series vs multi-column DataFrame
   - Required input columns (Close-only vs OHLCV)
2. Implement the indicator:
   - Add a new file under `trading-bot/src/trading_bot/indicators/` following the `IndicatorBase` pattern.
   - Do not mutate the input DataFrame; return new series/frame.
3. Wire into config validation (if it will be selectable via config):
   - Add the lowercase indicator name to `trading-bot/src/trading_bot/models/config.py` (`ALLOWED_INDICATORS`).
4. Add tests:
   - Extend `trading-bot/tests/test_indicators.py` with a basic shape/NaN sanity test.
5. If a strategy uses it, update that strategy’s `_add_indicators()` path and keep it look-ahead safe (shift where appropriate).

## Output requirements

- List files added/changed.
- Explain any shifting/lagging decisions to prevent look-ahead.
