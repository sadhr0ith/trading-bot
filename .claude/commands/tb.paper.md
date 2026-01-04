---
description: Prepare a safe paper-trading run command (single or multi-strategy) and only execute it when explicitly asked.
---

## User Input

```text
$ARGUMENTS
```

## Safety rules (must follow)

- This repo is paper trading only; never add real execution paths.
- The paper trading loop is long-running. Do **not** start it unless the user explicitly asks (e.g. `$ARGUMENTS` contains `run` or `--run`).
- Paper trading writes state to disk. Do not delete/reset state unless the user explicitly asks.

## Flow

1. Ensure we are in `trading-bot/` (the Python package root) and the environment is set up.
2. Parse `$ARGUMENTS`:
   - If it contains `--strategy <name>` → single strategy mode.
   - If it contains `--strategies a,b,c` or a comma-separated list → multi-strategy mode.
   - If empty/unclear → ask which strategy/strategies to run and which ticker/interval/period config to use.
3. Build the exact command to run:
   - Single: `python -m trading_bot.main --strategy <strategy>`
   - Multi: `python -m trading_bot.main --strategies <s1,s2,...>`
4. Explain what will happen:
   - Single strategy uses `paper_trading_state_<strategy>.json`
   - Multi-strategy uses shared `paper_trading_state.json`
   - ML strategies will be forced into `inference_only=True` during runtime; training is via `python -m trading_bot.train`
5. If `$ARGUMENTS` contains `run` or `--run`, execute the command. Otherwise, only print the command and ask for confirmation.

## Output requirements

- Print the final command exactly as it should be run.
- Mention which state file will be used and where (`TRADING_BOT_STATE_DIR` if set; otherwise current directory).
