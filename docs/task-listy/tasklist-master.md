# Trading Bot – Master Tasklist (merged)
**Ostatnia aktualizacja:** 2025-12-03  
**Źródła scalone:** `tasklist-master2.md`, `tasklist-sklearn-refactor.md`, `tasklist-todo.md`  
**Legenda priorytetów:** 🔴 P0 (blokery), 🟠 P1 (kluczowe), 🟡 P2 (utrzymanie/rozszerzenia)

---

## ✅ Zrealizowane (z datą)
- 2025-12-03: 🔴 Pydantic configy (StrategyConfig/RiskConfig), BaseSettings dla ENV (Binance/Gmail/Cache/Logging), walidacja metadata persistencji i state paper trading; drift check w `main.py`; dokument `docs/config-schema.md`.
- 2024-12-25: 🔴 `.env` + `.gitignore`; klucze Binance/Gmail z ENV; przykład `.env.example`.
- 2024-12-25: 🟠 `requirements.txt` doprecyzowane (scikeras, python-binance, pytest + piny).
- 2024-12-25: 🟡 Logger respektuje `log_level` (config/ENV); brak duplikacji handlerów.
- 2024-12-25: 🔴 `parse_period_to_timedelta` z obsługą `y` + testy jednostkowe.
- 2024-12-25: 🔴 Wskaźniki: `sma.py`, `ema.py`, `stochastic.py`, `bollinger_bands.py`; configi bez `fundamental_analysis`.
- 2024-12-25: 🟠 `data_fetcher`: klucze z ENV, bez `print`, logowanie; backoff w `main.py` dla pustych danych.
- 2024-12-25: 🟠 `email_notifications`: walidacja subject/body, lista odbiorców, obsługa błędów.
- 2024-12-25: 🟠 Strategie short/mid: dodane cechy lag/vol/SMA i forward target (5d/20d); brak pipeline’ów.
- 2025-12-03: 🟠 Usunięto zduplikowane cechy mid/long, dodano logowanie importancji, baseline dla LSTM, sanity na inference (short), retry/QA w fetcherze Binance, rozszerzone featury day-trading (Volume/volatility), dokumentacja modeli.
- 2025-12-03: 🟡 Caching danych (TTL), fallback ElasticNet dla małych próbek short-term, ATR dla LSTM, statystyki cech inference, dodatkowe testy integracyjne i troubleshooting w dokumentacji.

## Historia wykonanych zadań (timeline)
- 2024-12-25:
  - Dodane `.env`, `.gitignore`, przykład `.env.example`; klucze Binance/Gmail z ENV.
  - Uporządkowane zależności (`requirements.txt`), logger respektujący `log_level`.
  - Pierwsze wskaźniki (SMA, EMA, Stochastic, Bollinger), walidacja periodów, backoff w `main.py`, walidacja maili.
  - Początkowe featury short/mid (lagi/vol/SMA) bez pipeline’ów.
- 2025-12-02:
  - Pełne pipeline’y short/mid/long + LSTM (sliding window), persistencja artefaktów z metadata.
  - Hyperparam tuning (RandomizedSearchCV short, Halving mid), baseline ElasticNet/Naive, centralny seeding.
  - RiskManager + PaperTradingExecutor wpięte we wszystkie strategie; naprawa cashflow i izolacji state; indeks Binance → DatetimeIndex.
  - Parser periodów do `utils.time_utils`; short-term inference na 25+ wierszach; ujednolicone logowanie.
  - ADX + cechy kalendarzowe; długie lagi mid/long; testy: transformatory, pipeline, risk+paper, validatory, parse_period.
- 2025-12-03:
  - Dedup/ pruning cech mid/long, log top importances i statystyk inference; baseline LSTM (ostatni close).
  - Short-term: lekki tuner przy małych próbkach, fallback ElasticNet dla małych datasetów, log statystyk inference.
  - Day-trading: nowe featury Volume/Volatility/ATR, log statystyk okna, baseline MAE.
  - Fetcher Binance: retry/backoff, QA metryk (NaN, luki), caching TTL (env) + `cache/*.pkl`.
  - Testy integracyjne (pipeline save/load z DummyRegressor, stop-loss przy HOLD, zero balance), fixtures syntetyczne.
  - Dokumentacja: architektura z uzupełnioną tabelą strategii, rationale modeli, troubleshooting, state naming; tasklisty odświeżone.
- 2025-12-03 (Pydantic):
  - StrategyConfig/RiskConfig w Pydantic + ConfigValidator jako wrapper.
  - BaseSettings dla Binance/Gmail/cache/log level; wpięcie w `data_fetcher` i `email_notifications`.
  - Walidacja metadata persistencji (PersistenceMetadata) i pliku stanu paper trading (PaperState).
  - CLI: choices dla `--strategy`, logowanie driftu configu vs. metadata; dokument `docs/config-schema.md`; nowe testy `test_pydantic_config.py`.

---

## 🚧 P0 – blokery (do zrobienia)
- [X] (2025-12-02) Short-term: pełny `Pipeline` sklearn (transformery FE + `StandardScaler`) z `XGBRegressor`, walidacja czasowa (`TimeSeriesSplit`/walk-forward), brak ręcznego FE.
- [X] (2025-12-02) Short-term: persistencja całego pipeline’u (`joblib`) i inference wyłącznie przez zapisany pipeline.
- [X] (2025-12-02) Day-trading (LSTM): sliding windows (seq_len ~60), skalowanie tylko train, `TimeSeriesSplit`, brak przecieków; zapis/wczytanie modelu + scaler_X + scaler_y + metadata; inference na ostatnim oknie.
- [X] (2025-12-02) Mid-term: pipeline z transformerami (lagi/BB/MACD) + `TimeSeriesSplit`, persistencja.
- [X] (2025-12-02) Integracja `RiskManager` + `PaperTradingExecutor` w strategiach (BUY/SELL/HOLD → egzekucja, SL/TP, sizing, historia).

## 🟠 P1 – jakość i reprodukowalność
- [X] (2025-12-02) Hyperparam tuning (`RandomizedSearchCV` short-term / `HalvingRandomSearchCV` mid-term) z `TimeSeriesSplit`; LSTM walidacja na oknie val wciąż do zrobienia.
- [X] (2025-12-02) Baseline’y per horyzont (Naive, ElasticNet) + porównanie metryk.
- [X] (2025-12-02) Hyperparam tuning LSTM: walidacja na oknie val.
- [X] (2025-12-02) Ujednolicenie nazw kolumn wskaźników + walidator kolumn (`MACD`, `Signal`, `MACD_Histogram`, `RSI`, `BB_Upper/Middle/Lower/Width`, `Stochastic_K/D`).
- [X] (2025-12-02) Centralne seedy (numpy/random/xgboost/tf) – jeden helper wywoływany w strategiach.
- [X] (2025-12-02) Persistencja modeli: katalog per strategia, metadata (cechy, parametry, wersja, timestamp); ładowanie najnowszego modelu zamiast retrain w pętli.
- [X] (2025-12-03) Short-term inference okno 25+ wierszy z logiem NaN, lekki tuner przy małych próbkach; baseline LSTM (ostatni close) w day-trading; DateTimeIndex kalendarz na Binance.
- [X] (2025-12-03) Caching danych (env TTL), fallback ElasticNet dla małych próbek short-term, log statystyk cech inference/train, ATR/vol/volume w LSTM.

## 🟡 P2 – rozszerzenia i porządki
- [X] (2025-12-02) Dodanie ADX i cech kalendarzowych (day_of_week, month, month_start/end) do transformera short-term.
- [X] (2025-12-02) Dłuższe lagi/okna dla mid/long-term (20/60/120/250) + momentum/vol/drawdown w transformerach.
- [X] (2025-12-02) Testy: transformatory, pipeline fit/predict, integracja risk+paper, validatory kolumn, rozszerzony `parse_period`.
- [X] (2025-12-02) Dokumentacja: schemat pipeline’ów (train/predict), konwencje nazw kolumn, procedura persistencji/ładowania.
- [X] (2025-12-02) Cleanup: martwe importy/duplikacje; `check_is_fitted` lub guardy w modelach/pipeline’ach.
- [X] (2025-12-03) Redukcja cech wysokokorelacyjnych mid/long + log importancji; sanity/statystyki inference; retry/backoff i QA metryk w fetcherze Binance; doprecyzowane modele w architekturze.
- [X] (2025-12-03) Testy integracyjne (pipeline z DummyRegressor, stop-loss HOLD, zero balance), troubleshooting i dokumentacja state naming; caching opcjonalny.

---

## 📋 Post-Refactor Review & Bugfix Backlog (2025-12-02)

**Status refactoringu:** ✅ Główny refactoring pipeline'ów sklearn/LSTM zakończony
**Źródło:** Code review w `merged-after-refactor-overview.md` + weryfikacja kodu
**Szczegółowa tasklista:** Zobacz `docs/tasklist-bugfix-after-refactor.md` (z konkretnymi linijkami kodu i szczegółami technicznymi)

### Zidentyfikowane problemy wymagające naprawy:

**🔴 P0 – Must-Fix Blockers (6 zadań, ~12-15h):**
1. **Paper trading cash accounting** – linia 88 odejmuje tylko fee, linia 48 dodaje tylko PnL (method: `process_signal()`)
2. **Period parsing** – parser istnieje w data_fetcher.py (linia 13-31), trzeba przenieść do utils/time_utils.py
3. **Binance data index** – kolumna 'Open time' konwertowana (linia 72) ale nie ustawiona jako index
4. **Short-term inference** – linia 51 używa 1 wiersza, potrzebuje 25 dla lag/rolling features
5. **Hardcoded email** – linia 36 ma "sadhroith@gmail.com" jako fallback
6. **Risk config keys** – test używa `stop_loss_pct` (linia 8), RiskManager oczekuje `stop_loss`

**🟠 P1 – Secondary Issues (4 zadania, ~6-9h):**
7. Shared paper trading state path – leak między strategiami
8. Duplicate features – redundancja w mid/long-term (return/momentum)
9. Unused/dead code – model wrappers, backtesting.py nie są zintegrowane
10. Logging consistency – data_fetcher.py używa raw `getLogger`

**🟡 P2 – Improvements (3 zadania, OPTIONAL):**
11. Binance fetcher robustness – retry logic, validation (defer)
12. Enhanced test coverage – niektóre jako SKIP (stress test 1M+, property-based)
13. Documentation updates – większość w ramach P0/P1 tasks (defer resztę)

**Plan implementacji (ZREWIDOWANY):**
- **Faza 1 (2-3 dni):** P0 Critical – **#1 (START HERE)**, #5, #3, #2
- **Faza 2 (2-3 dni):** P0 Core + P1 Start – #4, #6, #7
- **Faza 3 (1-2 dni):** P1 Cleanup – #10, #8, #9
- **Faza 4 (Optional):** P2 Enhancements – defer większość, skip niektóre
- **Całość P0+P1: 20-25h / 3-4 dni focused work**

### Podsumowanie ukończonego refactoringu (z poprzednich tasklist):
- ✅ Short-term: pełny sklearn Pipeline z transformerami + XGBoost
- ✅ Day-trading: LSTM z sliding windows + proper train/val split
- ✅ Mid-term: pipeline z transformerami + TimeSeriesSplit
- ✅ Persistencja: joblib dla całych pipeline'ów + metadata
- ✅ Integracja: RiskManager + PaperTradingExecutor w strategiach
- ✅ Hyperparam tuning: RandomizedSearchCV / HalvingRandomSearchCV
- ✅ Baseline'y: Naive, ElasticNet per horyzont
- ✅ Naming convention: ujednolicone nazwy kolumn wskaźników
- ✅ Seeding: centralny helper dla reprodukowalności
- ✅ Feature engineering: ADX, calendar features, długie lagi dla mid/long-term
- ✅ Testy: transformatory, pipeline, risk+paper integration, validatory
- ✅ Dokumentacja: schemat pipeline'ów, konwencje, procedury persistencji

---

## 📝 Notatki wykonawcze
- FE do `utils/transformers.py` (LagFeatureTransformer, RollingStatsTransformer, IndicatorTransformer, PriceFeatureTransformer); pipeline = ColumnTransformer + model.
- Inference zawsze przez `.predict` zapisanego pipeline’u (`joblib`); zero ręcznych scalerów/fillna(0).
- LSTM: helper do okien i persistencji; alternatywa LightGBM na lagach jako baseline.
- Risk/paper: strategie powinny zwracać sygnał → executor + risk manager decyduje o pozycji/wyjściu; zapis historii trades/balance.
