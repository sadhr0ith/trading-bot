# Tasklist – Model & Feature Improvements
**Data utworzenia:** 2025-12-02  
**Zakres:** Poprawa doboru algorytmów, cech i walidacji w strategiach
**Legenda:** 🔴 P0 (must-fix), 🟠 P1 (ważne), 🟡 P2 (nice-to-have)

---

## 🔴 P0 – Krytyczne wyrównanie train/inference i baseliny
- [ ] **Spójne okno inference w short-term** (`trading-bot/strategies/short_term_strategy.py`)  
  - Zamiast pojedynczego wiersza użyj ostatnich N≥max(lag, rolling) (min. 25) przy predykcji.  
  - Dodaj walidację długości danych i ostrzeżenia, gdy cechy mają NaN po FE.
- [ ] **Prosty baseline w każdej strategii**  
  - Dodaj/utrzymaj ElasticNet jako baseline metrykę w short/mid/long; loguj MAE.  
  - W day-trading przygotuj prosty AR/EMA baseline (np. EMA przewidująca następny close) do porównań z LSTM.
- [ ] **Ujednolicenie kalendarzowych cech**  
  - Zapewnij, że dane z Binance mają `DatetimeIndex` przed FE.  
  - Zweryfikuj `CalendarFeatureTransformer` na wszystkich ścieżkach (Yahoo/Binance).

## 🟠 P1 – Dobór modeli i redukcja szumu
- [ ] **Short-term: sanity check modelu XGBoost**  
  - Porównaj z lżejszym modelem (ElasticNet/LightGBM jeśli dostępne) na tym samym FE.  
  - Ustal małą siatkę hyperparamów; ogranicz `n_estimators` / głębokość dla małych próbek.
- [ ] **Mid/Long-term: usuń duplikaty cech** (`strategies/mid_term_strategy.py`, `strategies/long_term_strategy.py`)  
  - Zidentyfikuj cechy wyliczane podwójnie (return vs momentum) na tych samych lagach i usuń duplikaty.  
  - Po redukcji porównaj MAE/feature_importances_.
- [ ] **Day-trading LSTM: wzbogacenie cech** (`strategies/day_trading_strategy.py`)  
  - Dodaj wolumen/volatility/RSI/ATR jako wejścia; upewnij się, że skaler jest dopasowany na tych samych kolumnach.  
  - Sprawdź, czy `SEQ_LEN` jest adekwatny do interwału (np. 60×1h = 2.5 dnia; ewentualnie skrócić lub wydłużyć zależnie od danych).

## 🟡 P2 – Walidacja i stabilność
- [ ] **Walk-forward tuning z małą siatką**  
  - Ustal stałą, małą liczbę prób w RandomizedSearch/Halving, aby uniknąć niestabilności przy małej próbce.  
  - Loguj parametry i wynik cv dla reprodukcji.
- [ ] **Monitorowanie dystrybucji cech**  
  - Dodaj krótki raport (np. log) z podstawowymi statystykami cech po FE (min/median/max, %NaN) dla train i inference.  
  - Ostrzegaj, gdy rozkład znacząco odbiega (np. brak wolumenu).
- [ ] **Dokumentacja wyboru modeli**  
  - Krótki opis w `docs/architecture.md` (lub osobny plik) dlaczego dany model dla horyzontu (LSTM day, XGB short, RF/GBM mid/long) oraz jakie baseliny są sprawdzane.
