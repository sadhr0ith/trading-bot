# Tasklist – Pydantic Improvements
**Data utworzenia:** 2025-12-03  
**Zakres:** Ujednolicenie walidacji konfiguracji i ustawień środowiskowych przy pomocy Pydantic  
**Legenda:** 🔴 P0 (blokery), 🟠 P1 (ważne), 🟡 P2 (nice-to-have)

---

## 🔴 P0 – Krytyczne
- [x] **Modele konfiguracji strategii w Pydantic** (`trading-bot/configs/*.py`, nowy moduł np. `models/config.py`)  
  - BaseModel z polami: `strategy` (Literal), `data_source` (Literal), `ticker` (str), `period`/`interval` (str, walidacja formatu), `indicators` (Set[Literal]), `use_indicators` (bool), `log_level` (Literal/enum), `notification_email` (EmailStr | List[EmailStr]), `seed` (int), `risk_management` (embedded).
  - Walidacje krzyżowe: gdy `data_source='binance'` wymagaj interwałów obsługiwanych w `data_fetcher`, gdy `use_indicators=False` wymuś pustą listę `indicators`.
  - Dostarcz sensowne domyślne wartości (np. `use_indicators=True`, pusta lista maili).
- [x] **RiskConfig z twardymi constraintami** (`utils/risk_management.py`)  
  - Pydantic model dla `stop_loss`, `take_profit`, `max_position_size`, `trading_fee` z zakresem [0,1]; walidacja brak nieznanych kluczy.  
  - Integracja z `RiskManager`: konstruktor przyjmuje `RiskConfig`, usuń ręczne parsowanie i ostrzeżenia o nieznanych polach.
- [x] **Refactor `config_handler.load_config` na Pydantic** (`config_handler.py`)  
  - Ładuj dict z pliku, parsuj do `StrategyConfig` (konkretny model per strategy lub union).  
  - Fail-fast: jeśli walidacja się nie powiedzie, loguj listę błędów Pydantic i zwracaj `None`.
  - Usuń/odchudź `ConfigValidator` (zostaw cienki wrapper do logowania?).
- [x] **Testy walidacji konfiguracji** (`tests/test_validators.py` lub nowy `tests/test_pydantic_config.py`)  
  - Scenariusze: brak wymaganych pól, zły `data_source`, wskaźnik spoza listy, zły email, wartości `stop_loss > 1` lub `<0`, interwał spoza listy Binance/Yahoo, `indicators` niespójne z `use_indicators=False`.  
  - Sprawdź, że zwracany jest klarowny błąd Pydantic i brak crasha w `run_trading_bot`.

## 🟠 P1 – Ważne
- [x] **Centralne ustawienia środowiskowe w BaseSettings** (`utils/env_settings.py` nowy)  
  - Modele: `BinanceSettings` (klucze obowiązkowe), `EmailSettings` (GMAIL_*), `CacheSettings` (TTL int >=0), `LoggingSettings` (poziom).  
  - Użycie w `data_fetcher` (Binance/API), `email_notifications` (Gmail), cache TTL; walidacja pustych/niewłaściwych wartości.
- [x] **Normalizacja i typowanie metadata persistencji** (`utils/model_persistence.py`)  
  - Pydantic model dla metadata (feature_columns: list[str], trained_until: str, mae_cv: Optional[float], config_signature: str, version: str, saved_at: datetime).  
  - Walidacja przy zapisie/odczycie, logowanie brakujących pól.
- [x] **Schematy Pydantic dla stanów paper trading** (`utils/paper_trading.py`)  
  - Modele: `PositionState`, `TradeRecord`, `PaperState`; walidacja podczas odczytu pliku JSON (wyłapywanie uszkodzonego stanu).  
  - Migration path: jeśli walidacja nie przejdzie, loguj ostrzeżenie i twórz stan początkowy.

## 🟡 P2 – Nice-to-have
- [x] **Eksport JSON Schema i dokumentacja** (`docs/architecture.md` lub nowy `docs/config-schema.md`)  
  - Wygeneruj schema dla `StrategyConfig`/`RiskConfig`/`EnvSettings`; dodaj przykładowy plik konfiguracyjny i mapowanie na istniejące `configs/*.py`.
- [x] **CLI validation & autocompletion** (`main.py`)  
  - Użyj Pydantic do walidacji argumentu `--strategy` (enum) oraz opcjonalnych flag (np. ścieżka configu); wyrzuć klarowny komunikat zamiast `ValueError`.
- [x] **Monitoring spójności runtime**  
  - Lekki check w pętli `run_trading_bot`: hash configu vs. metadata modelu z Pydantic (drift detection), ostrzeżenie gdy różne.

---

**Notatka:** Priorytetem jest zastąpienie ręcznych walidatorów konfiguracji/risk Pydantikiem i integracja z loaderem, tak aby błędy konfiguracyjne były wychwytywane przed uruchomieniem pętli bota.
