# Trading Bot – Bugfix Tasklist (Post-Refactor)
**Data utworzenia:** 2025-12-02
**Ostatnia rewizja:** 2025-12-02 (technical details verified from code)
**Źródło:** `merged-after-refactor-overview.md` review + code inspection
**Legenda priorytetów:** 🔴 P0 (must-fix blokery), 🟠 P1 (secondary issues), 🟡 P2 (optional/nice-to-have)

**Quick Summary:**
- **6 Critical Blockers (P0)** - must fix, ~12-15h total
- **4 Secondary Issues (P1)** - should fix, ~6-9h total
- **3 Optional Enhancements (P2)** - nice-to-have, defer
- **Estimated P0+P1 total: 20-25 hours / 3-4 days focused work**
- **Priority #1: Task #1 (Paper Trading) - START HERE**

---

## 🔴 P0 – Must-Fix Blockers

### 1. Paper Trading Cash Accounting Fix [CRITICAL - START HERE]
**Problem:** BUY subtracts only fee (line 88), not notional; SELL adds only PnL (line 48) without returning principal
**Plik:** `trading-bot/utils/paper_trading.py`
**Metoda:** `PaperTradingExecutor.process_signal()` lines 64-121
**Impact:** Balances/PnL are completely wrong, paper trading unusable

**Obecna implementacja (BŁĘDNA):**
```python
# Line 88 - BUY: odejmuje tylko FEE, nie notional!
self.state["balance"] -= price * size * risk_manager.trading_fee

# Line 48 - SELL: dodaje tylko PnL, nie zwraca notional!
pnl = (price - entry_price) * size
self.state["balance"] += pnl
```

**Szczegóły implementacji:**
- [x] **1.1** Fix line 88 (BUY flow):
  ```python
  notional = price * size
  fee = notional * risk_manager.trading_fee
  self.state["balance"] -= notional + fee
  ```
  - Odejmij notional (price × size) + fee
  - **Zapamiętaj notional w stanie pozycji:** `self.state["positions"][symbol]["entry_notional"] = notional`
- [x] **1.2** Fix line 48 (_close_position SELL flow):
  ```python
  notional = price * size
  fee = notional * risk_manager.trading_fee
  self.state["balance"] += notional - fee
  # PnL dla history (net after fees):
  entry_fee = position.get("entry_notional", entry_price * size) * risk_manager.trading_fee
  pnl = (price - entry_price) * size - entry_fee - fee
  ```
  - Dodaj notional zamknięcia - fee zamknięcia
  - PnL oblicz jako net po obu fees (entry + exit)
- [x] **1.3** Dodaj walidację: po BUY sprawdź `if self.state["balance"] < 0`, rollback i return error
- [x] **1.4** Test jednostkowy: BUY 1 BTC @ 50k with fee 0.001
  - Initial: 100k
  - Notional: 50k, Fee: 50k × 0.001 = 50
  - Expected balance: 100k - 50k - 50 = 49,950 ✅
- [x] **1.5** Test jednostkowy: SELL 1 BTC @ 55k with fee 0.001
  - Balance before: 49,950
  - Notional: 55k, Fee: 55k × 0.001 = 55
  - Expected balance: 49,950 + 55k - 55 = 104,895 ✅
  - Expected PnL in history: (55k - 50k) - 50 - 55 = 4,895 ✅
- [x] **1.6** Test integracyjny: full cycle BUY@50k→SELL@55k
  - Initial: 100k
  - After BUY: 49,950
  - After SELL: 104,895
  - Net profit: 4,895 (5k gain - 105 total fees) ✅
- [x] **1.7** Test edge case: BUY with insufficient balance should fail gracefully with clear error message

---

### 2. Period Parsing & Time Utils Module
**Problem:** Parser exists in `data_fetcher.py` (lines 13-31) but should be in `utils.time_utils`, uses `mo` for months (confusing), lacks `w` for weeks
**Pliki:** `trading-bot/data_fetcher.py`, `tests/test_parse_period_extended.py` (imports missing module)
**Impact:** Tests fail, period parsing is confusing and not reusable

**Obecna implementacja:**
```python
# data_fetcher.py lines 13-31 - exists but in wrong place!
def parse_period_to_timedelta(period):
    # Uses: 'mo' = months, 'y' = years, 'd' = days, 'h' = hours, 'm' = minutes
    # Missing: 'w' for weeks
```

**Szczegóły implementacji:**
- [x] **2.1** Create `trading-bot/utils/time_utils.py`
- [x] **2.2** Move `parse_period_to_timedelta()` from data_fetcher.py to time_utils.py
- [x] **2.3** **CHOOSE CONVENTION (implement Option A):**
  - **Option A (RECOMMENDED - implement this):** `m`=minutes, `M`=months, `w`=weeks, `d`=days, `h`=hours, `y`=years
  - ~~Option B: `m`=months, `w`=weeks, `d`=days, `h`=hours, `min`=minutes, `y`=years~~
  - **Rationale:** More standard (M/m distinction is common in finance/pandas)
- [x] **2.4** Add support for `w` (weeks) = 7 days
- [x] **2.5** Update `data_fetcher.py` line 99: import from `utils.time_utils`
- [x] **2.6** Create `tests/test_time_utils.py` with comprehensive tests:
  - Valid periods per chosen convention: `1m`, `5m`, `1h`, `1d`, `1w`, `1M`, `6M`, `1y`
  - Invalid formats: `1x`, `abc`, `-1d`, `1.5d` (tests already exist in test_data_fetcher.py)
- [x] **2.7** Fix `tests/test_parse_period_extended.py` import statement
- [x] **2.8** Document period format in docstring with examples + **link to Task #13.1 for README update**

**Note:** DO NOT create `parse_period_to_binance_interval()` - interval already comes from config (e.g., "1d", "4h")

---

### 3. Binance Data Index (Numeric → DateTime)
**Problem:** 'Open time' is converted to datetime (line 72) but NOT set as index → DataFrame has RangeIndex
**Plik:** `trading-bot/data_fetcher.py`, function `fetch_binance_data()` lines 47-81
**Impact:** Calendar features computed on numeric index instead of time, breaks time-based operations

**Obecna implementacja:**
```python
# Line 72-74 - converts but doesn't set as index!
data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
data['Close time'] = pd.to_datetime(data['Close time'], unit='ms')
# Missing: data.set_index('Open time', inplace=True)
```

**Szczegóły implementacji:**
- [x] **3.1** Add line after 74 in `data_fetcher.py`: `data.set_index('Open time', inplace=True)`
- [x] **3.2** Verify index name: optionally rename to lowercase: `data.index.name = 'timestamp'`
- [x] **3.3** Add assertion after setting index: `assert isinstance(data.index, pd.DatetimeIndex)`
- [x] **3.4** Test calendar features: verify `CalendarFeatureTransformer` works with DatetimeIndex
  - day_of_week should be 0-6, not random numbers
  - month should be 1-12 (already has proper DatetimeIndex handling)
- [x] **3.5** Add test in `tests/test_data_fetcher.py`:
  - Mock Binance client response
  - Verify returned DataFrame has DatetimeIndex
  - Verify index is monotonically increasing
- [x] **3.6** Check all strategies handle DatetimeIndex correctly (should be transparent)
- [x] **3.7** Verify no hardcoded numeric index assumptions in transformers

---

### 4. Short-Term Inference Feature Engineering Fix
**Problem:** Line 51 uses single-row DataFrame for inference → lag/rolling transformers produce NaN
**Plik:** `trading-bot/strategies/short_term_strategy.py`, method `execute()` line 51
**Impact:** All lag features become NaN→0 in inference, predictions completely wrong

**Obecna implementacja:**
```python
# Line 51 - only last row, no history for lags!
data_inference = data.iloc[[-1]].drop(columns=['target'])

# But pipeline needs history for:
# - LagFeatureTransformer: lags=[1, 3, 5, 10] → needs 10 previous rows
# - RollingStatsTransformer: windows=[5, 10] → needs 10 previous rows
# Minimum required: 10 rows before the prediction row
```

**Szczegóły implementacji:**
- [x] **4.1** Calculate minimum window: `max(lags) + max(windows) = 10 + 10 = 20` rows (with safety margin: 25)
- [x] **4.2** Change line 51 to: `data_inference = data.iloc[-25:]` (last 25 rows including target row)
- [x] **4.3** Keep only last prediction: `predicted_return = float(pipeline_for_inference.predict(data_inference)[-1])`
- [x] **4.4** Add validation before line 51: ensure `len(data) >= 25`, else log error and return
- [x] **4.5** Add conditional debug logging for NaN detection (only in debug mode, avoid performance hit):
  ```python
  # After line 203 (before predict) - only if logger is DEBUG level
  if self.logger.isEnabledFor(logging.DEBUG):
      X_inference = pipeline_for_inference.named_steps['feature_engineering'].transform(data_inference)
      if np.isnan(X_inference[-1]).any():
          self.logger.warning(f"NaN features in inference: {np.where(np.isnan(X_inference[-1]))[0]}")
  ```
- [x] **4.6** Test with synthetic data:
  - Create DataFrame with 30 rows, known pattern (e.g., linear increase)
  - Train pipeline, then predict on last 25 rows
  - Verify lag features match expected values (e.g., Close_lag_1 == previous Close)
- [x] **4.7** Test edge case: data with exactly 25 rows → should work
- [x] **4.8** Test edge case: data with <25 rows → should fail gracefully with clear error
- [x] **4.9** Update docstring: document minimum data requirement (25+ rows)
- [x] **4.10** Apply same fix to other strategies if they have similar issue (mid_term, long_term - NO ISSUE FOUND)

---

### 5. Hardcoded Email Fallback Removal [QUICK WIN - 30min]
**Problem:** Line 36 has hardcoded fallback email "sadhroith@gmail.com" - security risk!
**Plik:** `trading-bot/utils/email_notifications.py`, function `send_email()` line 36
**Impact:** Leaks real email address, doesn't fail fast on missing config

**Obecna implementacja:**
```python
# Line 36 - HARDCODED REAL EMAIL ADDRESS!
from_email = os.getenv("GMAIL_SENDER_EMAIL", "sadhroith@gmail.com")
```

**Szczegóły implementacji:**
- [x] **5.1** Remove fallback from line 36: `from_email = os.getenv("GMAIL_SENDER_EMAIL")`
- [x] **5.2** Add validation after line 36:
  ```python
  if not from_email:
      logger.error("GMAIL_SENDER_EMAIL must be set in environment variables")
      return False
  ```
- [x] **5.3** Or combine into single check with GMAIL_APP_PASSWORD (lines 37-40)
- [x] **5.4** Update `.env.example` with clear comments:
  ```bash
  # Required for email notifications
  GMAIL_SENDER_EMAIL=your.email@gmail.com
  GMAIL_APP_PASSWORD=your_app_password_here
  ```
- [x] **5.5** Test: call `send_email()` without GMAIL_SENDER_EMAIL env var → should return False with clear error
- [x] **5.6** Test: call with valid credentials → should send successfully (tested with mocking)
- [x] **5.7** Add to documentation: how to generate Gmail App Password (added to .env.example comments)

---

### 6. Risk Config Keys Normalization
**Problem:** Test uses `stop_loss_pct`/`take_profit_pct` but RiskManager expects `stop_loss`/`take_profit` → keys silently ignored!
**Pliki:** `tests/test_risk_paper_integration.py` (line 8), `utils/risk_management.py` (lines 21-23)
**Impact:** Test stops/TP not applied, silent failure, wrong risk management

**Obecna implementacja:**
```python
# test_risk_paper_integration.py line 8 - uses *_pct:
rm = RiskManager({"max_position_size": 1000, "stop_loss_pct": 0.05, "take_profit_pct": 0.1})

# risk_management.py lines 21-22 - expects without _pct:
self.stop_loss = float(self.config.get("stop_loss", 0.0)) or None
self.take_profit = float(self.config.get("take_profit", 0.0)) or None
# → stop_loss_pct is IGNORED, stop_loss defaults to 0.0 → None!
```

**Current status:**
- ✅ Configs use correct keys: `stop_loss`, `take_profit`, `max_position_size` (config_short_term.py line 18-20)
- ❌ Test uses wrong keys: `stop_loss_pct`, `take_profit_pct` (test_risk_paper_integration.py line 8)
- ✅ RiskManager expects: `stop_loss`, `take_profit`, `max_position_size`

**Szczegóły implementacji:**
- [x] **6.1** Fix test line 8: change `"stop_loss_pct": 0.05` → `"stop_loss": 0.05"`
- [x] **6.2** Fix test line 8: change `"take_profit_pct": 0.1` → `"take_profit": 0.1"`
- [x] **6.3** Add validation to RiskManager.__init__() to warn on unknown keys:
  ```python
  valid_keys = {"stop_loss", "take_profit", "max_position_size", "trading_fee"}
  unknown_keys = set(self.config.keys()) - valid_keys
  if unknown_keys:
      self.logger.warning(f"Unknown risk config keys (will be ignored): {unknown_keys}")
  ```
- [x] **6.4** Grep for any other files using `*_pct` variants: `rg "stop_loss_pct|take_profit_pct"`
- [x] **6.5** Add test: pass config with unknown key → verify warning is logged
- [x] **6.6** Document risk config schema in RiskManager docstring:
  - stop_loss: float (e.g., 0.03 = 3% stop loss)
  - take_profit: float (e.g., 0.05 = 5% take profit)
  - max_position_size: float (e.g., 0.1 = 10% of balance)
  - trading_fee: float (e.g., 0.001 = 0.1% fee)
- [x] **6.7** Add example to docstring showing proper config format

---

## 🟠 P1 – Secondary Issues

### 7. Shared Paper Trading State Path
**Problem:** Default `paper_trading_state.json` causes position leakage between strategies
**Plik:** `trading-bot/utils/paper_trading.py`
**Impact:** One strategy's positions can leak into another run

**Szczegóły implementacji:**
- [x] **7.1** Add `state_file_path` parameter to `PaperTradingExecutor.__init__()`
- [x] **7.2** Generate default path per strategy: `paper_trading_state_{strategy_name}.json`
- [x] **7.3** Update all strategy instantiations to pass unique state file path
- [x] **7.4** Add `strategy_name` field to state JSON for verification
- [x] **7.5** Add validation: warn if loading state with different strategy_name
- [x] **7.6** Test: run two strategies concurrently, verify isolated states
- [x] **7.7** Documentation: Created comprehensive tests documenting state file naming convention

---

### 8. Duplicate/Low-Signal Features in Mid/Long-Term
**Problem:** Return and momentum duplicates across same lags inflate feature space
**Pliki:** `trading-bot/strategies/mid_term_strategy.py`, `trading-bot/strategies/long_term_strategy.py`
**Impact:** Model complexity without added signal, potential overfitting

**Szczegóły implementacji:**
- [x] **8.1** Audit features in `mid_term_strategy.py`: list all lag/momentum/return features
- [x] **8.2** Audit features in `long_term_strategy.py`: list all lag/momentum/return features
- [x] **8.3** Identify exact duplicates (same calculation, different names)
- [x] **8.4** Identify near-duplicates (e.g., return_5d and momentum_5d if calculated same way)
- [x] **8.5** Create correlation matrix of all features on sample data
- [x] **8.6** Remove features with correlation > 0.95 to existing features
- [x] **8.7** Consolidate transformers to avoid redundant calculations
- [x] **8.8** Run feature importance analysis (XGBoost feature_importances_)
- [x] **8.9** Remove features with consistently zero importance
- [x] **8.10** Document final feature set in strategy docstrings
- [x] **8.11** Compare model performance before/after feature reduction

---

### 9. Unused/Dead Code Cleanup
**Problem:** Model wrappers, backtesting.py, config `models` entries not wired into strategies
**Pliki:** `trading-bot/models/*.py`, `trading-bot/backtesting.py`, configs
**Impact:** Code maintenance burden, confusion

**Szczegóły implementacji:**
- [x] **9.1** Search for all imports of `trading-bot/models/` modules
- [x] **9.2** Search for all imports of `backtesting.py`
- [x] **9.3** Verify if `model_base.py`, `random_forest_model.py`, `xgboost_model.py` are used
- [x] **9.4** Decision point: DELETE or INTEGRATE?
  - If DELETE: remove files + remove from configs
  - If INTEGRATE: wire into strategies with proper pipeline integration
- [x] **9.5** Remove unused `models` config entries from all config files
- [x] **9.6** Search for other potential dead code (unused imports, functions)
- [x] **9.7** Run tests after removal to ensure nothing breaks
- [x] **9.8** Update documentation to reflect removed/integrated components

---

### 10. Logging Consistency
**Problem:** `data_fetcher.py` uses raw `logging.getLogger`, rest uses `utils.logger.setup_logger`
**Plik:** `trading-bot/data_fetcher.py`
**Impact:** Mixed formatting/levels, harder to debug

**Szczegóły implementacji:**
- [x] **10.1** Review `utils/logger.py` setup_logger implementation
- [x] **10.2** Replace `logging.getLogger(__name__)` in `data_fetcher.py` with `setup_logger(__name__)`
- [x] **10.3** Verify logger configuration is consistent (format, level, handlers)
- [x] **10.4** Grep for any other files using raw `logging.getLogger` (found config_handler.py, updated)
- [x] **10.5** Standardize all logging calls to use `utils.logger.setup_logger` (data_fetcher.py + config_handler.py)
- [x] **10.6** Logging style documented via comprehensive tests in test_logging_consistency.py
- [x] **10.7** Test: verify all logs have consistent format across modules (5 tests created and passing)

---

## 🟡 P2 – Improvements & Cleanup [OPTIONAL - Nice-to-Have]

### 11. Binance Data Fetcher Robustness [OPTIONAL]
**Problem:** Potential edge cases in data fetching
**Priority:** Low - current implementation works for happy path
**Szczegóły implementacji:**
- [x] **11.1** Add retry logic with exponential backoff for network failures (useful for production)
- [x] **11.2** Add validation: minimum required rows (e.g., >100 for training)
- [x] **11.3** Add validation: no gaps in timestamp sequence
- [x] **11.4** Add handling for Binance rate limits (weight tracking) - only needed for high-frequency usage
- [x] **11.5** Add caching layer (TTL, env-controlled, per source/ticker/period/interval)
- [x] **11.6** Log data quality metrics (missing values, outliers)

### 12. Enhanced Testing Coverage [OPTIONAL]
**Problem:** Need more comprehensive test coverage
**Priority:** Low - basic tests already exist, these are enhancements
**Szczegóły implementacji:**
- [x] **12.1** Integration test: full strategy pipeline train→save→load→predict (lightweight DummyRegressor)
- [x] **12.2** Integration test: paper trading full lifecycle (stop-loss exit on HOLD)
- [x] **12.3** Integration test: risk manager edge cases (zero balance → zero size)
- [x] **12.4** SKIP/Waived: Property-based tests - overkill for current scope
- [x] **12.5** SKIP/Waived: Stress test 1M+ rows - not realistic for trading bot
- [x] **12.6** Test fixtures/synthetic data for scenarios (bullish trend)

### 13. Documentation Updates [OPTIONAL]
**Problem:** Documentation should reflect fixes
**Priority:** Low - code comments and docstrings are more important
**Szczegóły implementacji:**
- [x] **13.1** Update README/docs with:
  - Correct config format examples (risk_management keys)
  - **Period format convention from Task #2:** `m`=minutes, `M`=months, `w`=weeks, `d`=days, `h`=hours, `y`=years
  - Examples: `"period": "1d"` (1 day), `"period": "6M"` (6 months), `"period": "1y"` (1 year)
- [x] **13.2** Document paper trading state file naming convention (covered in Task #7)
- [ ] **13.3** ~~Document period parsing format~~ (covered in Task #13.1)
- [ ] **13.4** ~~Document risk config schema~~ (covered in Task #6 docstrings)
- [x] **13.5** Add troubleshooting section for common errors
- [x] **13.6** SKIP/Waived: Architecture diagram - time-consuming, low value for now

### 14. Model & Feature Improvements (execute after P0/P1)
**Problem:** Usprawnienia algorytmów i cech po naprawie krytyków
**Priority:** Medium/Low – wykonać po P0/P1
**Szczegóły implementacji:**
- [x] **14.1** Short-term inference window alignment (`strategies/short_term_strategy.py`)  
  - Ustal N = max(lagów, rolling) z FE (min. 25); tnij dane do ostatnich N wierszy przed predykcją.  
  - Waliduj `len(data) >= N`; w przeciwnym razie loguj i zwracaj.  
  - Predykcja tylko z ostatniego wiersza po przejściu FE; log NaN tylko w trybie DEBUG.  
  - Testy: case 25 wierszy, <25 wierszy (graceful fail), synthetic pattern (sprawdź lags).
- [x] **14.2** Baseliny w strategiach  
  - Short/Mid/Long: ElasticNet baseline MAE logowany obok modelu głównego.  
  - Day-trading: prosty baseline (EMA/AR) prognozujący kolejny close; log porównania z LSTM.
- [x] **14.3** Kalendarzowe cechy na DateTimeIndex  
  - Upewnij się, że ścieżka Binance/Yahoo dostarcza DatetimeIndex przed FE.  
  - Przetestuj `CalendarFeatureTransformer` na obu źródłach (day_of_week, month, is_month_start/end).
- [x] **14.4** Short-term: sanity check XGBoost  
  - Porównaj z lżejszym modelem (ElasticNet/LightGBM jeżeli dostępne) na tym samym FE.  
  - Zmniejsz złożoność: ogranicz `max_depth`, `n_estimators`; mała siatka tuningu.
- [x] **14.5** Mid/Long-term: redukcja zduplikowanych cech  
  - Zidentyfikuj duplikaty (return vs momentum na tych samych lagach); usuń je.  
  - Correlation matrix >0.95 i feature_importances_ po redukcji; porównaj MAE.
- [x] **14.6** Day-trading LSTM: wzbogacenie cech (`strategies/day_trading_strategy.py`)  
  - Dodaj Volume, zmienność (rolling std), RSI, ATR jako wejścia; skaler obejmuje wszystkie kolumny.  
  - Zweryfikuj `SEQ_LEN` vs interwał (np. 60×1h=2.5d — dostosuj, jeśli potrzeba).
- [x] **14.7** Walidacja/tuning  
  - Mała stała siatka w RandomizedSearch/Halving; loguj paramy i wynik cv.  
  - Lekki raport statystyk cech (min/median/max, %NaN) train/inference dla szybkiego sanity checku.
- [x] **14.8** Dokumentacja wyboru modeli  
  - Dodaj do `docs/architecture.md` (lub osobny plik) rationale: który model dla jakiego horyzontu + używane baseliny.

---

## 📋 Implementation Order (Recommended)

**Phase 1 – Critical Blockers (Week 1 - 2-3 days):**
1. Task #1 (Paper trading accounting) – **START HERE** - blocker for all testing, 3-4h
2. Task #5 (Email fallback) – quick win, security, 30min
3. Task #3 (Binance index) – foundational for time features, 1-2h
4. Task #2 (Period parsing) – fix tests, improve clarity, 2h

**Note:** Tasks #2 and #3 are interchangeable - if tests are blocking, do #2 first; if time features need fixing, do #3 first. Both are quick (1-2h each).

**Phase 2 – Core Functionality (Week 1-2 - 2-3 days):**
5. Task #4 (Short-term inference) – model accuracy critical, 4-5h
6. Task #6 (Risk config) – prevent silent test failures, 1-2h
7. Task #7 (State path isolation) – testing reliability, 2h

**Phase 3 – Cleanup & Quality (Week 2 - 1-2 days):**
8. Task #10 (Logging consistency) – debugging, 1h
9. Task #8 (Feature deduplication) – model efficiency, 3-4h
10. Task #9 (Dead code cleanup) – maintenance, 2-3h

**Phase 4 – Enhancements (Week 3+ - ongoing, OPTIONAL):**
11. Task #11 (Binance robustness) – nice-to-have
12. Task #12 (Enhanced testing) – nice-to-have
13. Task #13 (Documentation) – nice-to-have

**Estimated total for P0+P1 (Phases 1-3): ~20-25 hours / 3-4 days focused work**

---

## 🎯 Definition of Done per Task
- [ ] Code implementation complete
- [ ] Unit tests written and passing
- [ ] Integration tests (if applicable) passing
- [ ] Documentation updated
- [ ] Code review completed
- [ ] No new warnings/errors in logs
- [ ] Performance impact assessed (if applicable)

---

## 📊 Progress Tracking
**Status:** 13/13 core tasks completed ✅ (ALL P0/P1 complete; optionals addressed, SKIP items explicitly waived)
- ✅ Task #1: Paper Trading Cash Accounting Fix (COMPLETED)
- ✅ Task #2: Period Parsing & Time Utils Module (COMPLETED)
- ✅ Task #3: Binance Data Index (COMPLETED)
- ✅ Task #5: Hardcoded Email Fallback Removal (COMPLETED)
- ✅ Task #6: Risk Config Keys Normalization (COMPLETED)
- ✅ Task #4: Short-Term Inference Feature Engineering Fix (COMPLETED - all 10 subtasks!)
- ✅ Task #7: Shared Paper Trading State Path (COMPLETED - all 7 subtasks!)
- ✅ Task #10: Logging Consistency (COMPLETED - all 7 subtasks!)
- ✅ Task #8: Duplicate/Low-Signal Features in Mid/Long-Term (COMPLETED)
- ✅ Task #9: Unused/Dead Code Cleanup (COMPLETED - no stray imports/backtesting remaining)
- ✅ Task #11: Binance robustness (retry, validation, caching)
- ✅ Task #12: Enhanced testing coverage (lightweight integration tests, synthetic fixtures)
- ✅ Task #13: Documentation updates (periods, risk keys, state naming, troubleshooting)
- ✅ Task #14: Model & feature improvements (sanity XGBoost fallback, expanded LSTM features, logging stats, rationale)
- ⚠️ SKIP/Waived: Task #12.4, #12.5, #13.6 (intentionally not implemented)

**Blockers:** None currently
**Phase 1 Status:** ✅ COMPLETE (all P0 quick wins)
**Phase 2 Status:** ✅ COMPLETE (Task #4 - largest P0 blocker)
**Phase 3 Status:** ✅ COMPLETE (P1 Tasks #7, #10, #8, #9 all done)
**Next Phase:** Optional SKIP items only (12.4/12.5/13.6) or further enhancements as desired
