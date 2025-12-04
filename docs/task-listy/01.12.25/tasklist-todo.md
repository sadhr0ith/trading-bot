# Trading Bot – Plan naprawczy (MVP-first)
**Status:** draft do zatwierdzenia  
**Cel:** stabilne MVP z poprawnym pipeline ML, walidacją danych i minimalnym backtestem/paper-tradingiem. Zaawansowane funkcje odkładamy.

## Fazy i priorytety
- 🔴 P0: blokery uruchomienia / ewidentne błędy
- 🟠 P1: kluczowe dla poprawności sygnałów
- 🟡 P2: jakości/utrzymanie
- 🟢 P3: nice-to-have/parking lot

## Phase 0 (1–2 dni) – sanity/setup
- [x] 🔴 Ujednolicenie nazewnictwa: `ticker` w configach; wskaźniki tylko te zaimplementowane (`rsi`, `macd`, `sma`, `ema`, `stochastic`, `bollinger_bands`); usunięcie `fundamental_analysis`.
- [x] 🔴 `.env` + `.gitignore`: dodać BINANCE_API_KEY/SECRET, GMAIL_APP_PASSWORD; brak sekretów w repo.
- [x] 🟠 `requirements.txt`: dodać scikeras, bez nowych modeli; wstępne pinowanie (pełne pinowanie później).
- [x] 🟡 Logger respektuje `log_level` z configu (przekazanie poziomu do setup_logger).
- [x] 🟡 Zdeprecjonować twardą zależność strategii od wskaźników: przygotować ścieżkę fallback na czyste cechy cenowe (lag/return/vol/MA) tak, by strategie działały bez MACD/RSI.

## Phase 1 (4–5 dni) – blokery
- [x] 🔴 Wskaźniki: dodać `sma.py`, `ema.py`, `stochastic.py` (K/D), `bollinger_bands.py` (kolumny: `BB_Upper/Middle/Lower/Width`), zgodnie z IndicatorBase.
- [ ] 🟠 Ujednolicić nazwy kolumn wskaźników w kodzie/strategiach (np. `MACD_Histogram` vs `Histogram`, `RSI` jako kolumna), aby uniknąć KeyError i niespójności.
- [x] 🔴 `parse_period_to_timedelta`: obsługa `y`; testy jednostkowe dla d/mo/y/h/m.
- [x] 🔴 Binance API keys z ENV + walidacja; zero `print` w data_fetcher, tylko logger.
- [x] 🟠 `email_notifications.py`: włączyć try/except, odbiorcy jako lista, walidacja subject/body.
- [x] 🟠 `main.py`: pętla w try/except z backoff; pomijanie trenowania, gdy `data` puste/None; logowanie błędów.

## Phase 2 (1–1.5 tyg.) – ML pipeline bez przecieków
- [x] 🔴 Day trading (LSTM): sliding window (seq_len ~60), shape (n, timesteps, features); TimeSeriesSplit; oddzielny val/test; skalowanie tylko train; target = przyszły Close/return; jawne drop/log NaN.
- [x] 🔴 Short term (XGBoost baseline): cechy temporalne (lagi 1/3/5/10, MACD/RSI lags, returns, rolling vol/MA), target = forward return (np. 5d, `shift(-5)`); split chronologiczny; predykcja na najnowszym wierszu, nie na test secie.
- [x] 🟠 Mid term (RF baseline): dodać wskaźniki z configu (MACD, Bollinger), lagi 5/10/20, target forward (20d return); brak „Close→Close”.
- [x] 🟠 Baseline/seed: dodać prosty baseline (naive/ElasticNet) i seedy (numpy/sklearn/tf) dla powtarzalności.
- [x] 🟠 Ujednolicenie interfejsu modeli: RF/XGB dziedziczą z ModelBase, wspólne `train/predict/evaluate`, jeden logger.
- [x] 🟠 Persistencja modeli na minimalnym poziomie: zapisz/wczytaj najnowszy model per strategia i unikaj pełnego retrainu w każdej iteracji pętli.
- [x] 🟠 Ścieżka price-only: wariant feature engineering bez wskaźników (lagged returns/prices, rolling vol/MA, range, volume) i porównanie metryk z wariantem z wskaźnikami; jeśli brak zysku → domyślnie wyłączyć wskaźniki.

## Phase 3 (1 tydzień) – walidacja/backtest/paper
- [x] 🔴 DataValidator: puste df, brak OHLCV, NaN/duplikaty, ceny ≤ 0, min_rows; errors/warnings.
- [x] 🟠 ConfigValidator: wymagane pola, wskaźniki tylko zaimplementowane; wywoływany w main przed startem.
- [x] 🟠 Minimalny backtester: walk-forward, metryki (total return, maxDD, win rate, Sharpe uproszczony), koszty transakcyjne stałe; bez wykresów.
- [x] 🟠 Paper-trading executor: prosty OrderExecutor (BUY/SELL, saldo, prowizja, zapis trade history), integracja ze strategiami; RiskManager minimalny (SL/TP, sizing uproszczony), monitorowanie pozycji w pętli.

## Phase 4 (1 tydzień) – refactor/tests
- [ ] 🔴 Testy jednostkowe (pytest): wskaźniki (RSI/MACD/SMA/EMA/Stoch/BB), DataValidator/ConfigValidator, proste testy strategii na sztucznych danych; mock zewnętrznych API; CI bez sieci.
- [ ] 🟠 Refactor duplikacji: wspólne helpery w StrategyBase (send_notification, split/scale); usunięcie powtarzanego kodu z 4 strategii.
- [ ] 🟠 Log rotation/structured logging (opcjonalnie po akceptacji).

## Parking lot (po MVP)
- LightGBM/CatBoost/Prophet/ARIMA, FeatureStore, ModelManager z wersjonowaniem, PerformanceMonitor/DB, wizualizacje backtestu, JSON logging, monitoring driftu.

## Uwagi do harmonogramu
- Fazy 0–2 to ~2 tyg., 3–4 kolejne ~2 tyg. Zaawansowane modele/monitoring dopiero po stabilnym MVP.
- Snippety kodu traktować jako szkic; utrzymać spójność nazw kolumn/wskaźników z istniejącym kodem.
