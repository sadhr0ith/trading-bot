# Trading Bot - Final Task List (Post Review)

**Data utworzenia:** 2025-12-04
**Źródło:** claude-final-review.md
**Total tasks:** 24 (5 BUGS + 5 P0 + 4 P1 + 10 P2)

**Zasady:**
- ✅ = Zrobione
- 🔄 = W trakcie
- ⏸️ = Zablokowane (czeka na inne zadanie)
- ❌ = Do zrobienia

---

## 🚨 SPRINT 0: CRITICAL BUGS (NATYCHMIAST - ~2h)

**UWAGA:** Te bugi mogą crashować production! Zrób przed jakąkolwiek inną pracą!

### BUG-1: DataValidator KeyError Fix
**Status:** ✅ Zrobione
**Priorytet:** 🔴 CRITICAL
**Effort:** 5 minut
**Plik:** `utils/validators.py`

**Problem:**
DataValidator kontynuuje walidację po znalezieniu brakujących kolumn, co prowadzi do KeyError gdy próbuje użyć kolumny która nie istnieje.

**Kroki:**
1. [ ] Otwórz `utils/validators.py`
2. [ ] Znajdź linię ~66 (po `errors.append(f"Missing required columns: {missing}")`)
3. [ ] Dodaj natychmiastowy return:
   ```python
   if self.require_ohlcv:
       missing = required_columns - set(df.columns)
       if missing:
           errors.append(f"Missing required columns: {missing}")
           return ValidationResult(
               is_valid=False,
               errors=errors,
               warnings=warnings,
               data=None
           )
   ```
4. [ ] Uruchom testy: `pytest tests/test_validators.py -v`
5. [ ] Dodaj test case dla missing columns:
   ```python
   def test_validator_missing_columns():
       """Test that validator returns early when columns missing."""
       df = pd.DataFrame({'Price': [100, 101]})  # Brak OHLCV
       validator = DataValidator(require_ohlcv=True)
       result = validator.validate(df)
       assert not result.is_valid
       assert "Missing required columns" in result.errors[0]
       assert result.data is None  # Should not crash
   ```
6. [ ] Commit: `git commit -m "Fix BUG-1: DataValidator KeyError on missing columns"`

**Verification:**
```bash
# Test manual:
python -c "
from utils.validators import DataValidator
import pandas as pd
df = pd.DataFrame({'Wrong': [1, 2]})
result = DataValidator(require_ohlcv=True).validate(df)
print('PASS' if not result.is_valid else 'FAIL')
"
```

---

### BUG-2: Podwójne Ładowanie Modeli
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 10 minut
**Pliki:**
- `strategies/short_term_strategy.py`
- `strategies/mid_term_strategy.py`

**Problem:**
Strategia ignoruje pipeline zwrócony przez `train_or_load_pipeline()` i ładuje ponownie z dysku.

**Kroki dla short_term_strategy.py:**
1. [ ] Otwórz `strategies/short_term_strategy.py`
2. [ ] Znajdź linię ~236 (`_pipeline = self.train_or_load_pipeline(...)`)
3. [ ] Zmień `_pipeline` na `pipeline` (usuń underscore)
4. [ ] Znajdź linię ~254 (gdzie ładuje ponownie z dysku)
5. [ ] Usuń cały blok reload:
   ```python
   # ❌ USUŃ TO:
   artifact = ModelPersistence().load(self.config["strategy"])
   if artifact:
       pipeline = artifact["pipeline"]

   # ✅ UŻYJ TEGO (już istnieje jako pipeline):
   # pipeline jest już załadowany/wytrenowany powyżej
   ```
6. [ ] Użyj `pipeline` bezpośrednio dla inference:
   ```python
   y_pred = pipeline.predict(X_inference)
   ```

**Kroki dla mid_term_strategy.py:**
7. [ ] Powtórz kroki 1-6 dla `strategies/mid_term_strategy.py` (~163-204)

**Testing:**
8. [ ] Dodaj timing test:
   ```python
   import time
   start = time.time()
   strategy.execute()
   duration = time.time() - start
   print(f"Execution time: {duration:.2f}s")  # Powinno być szybsze
   ```
9. [ ] Uruchom: `pytest tests/test_short_term_inference.py -v`
10. [ ] Commit: `git commit -m "Fix BUG-2: Remove duplicate model loading in short/mid-term"`

---

### BUG-3: Config Drift Detection Broken (day_trading)
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 5 minut
**Plik:** `strategies/day_trading_strategy.py`

**Problem:**
Day trading nie zapisuje `config_signature` w metadata, więc drift detection nigdy nie zadziała.

**Kroki:**
1. [ ] Otwórz `strategies/day_trading_strategy.py`
2. [ ] Na początku pliku dodaj import:
   ```python
   from utils.strategy_helpers import _build_config_signature
   ```
3. [ ] Znajdź linię ~230 (przed `persistence.save()`)
4. [ ] Dodaj obliczenie signature:
   ```python
   # Calculate config signature for drift detection
   config_blob = self.config.model_dump() if hasattr(self.config, "model_dump") else self.config
   config_signature = _build_config_signature(config_blob)
   ```
5. [ ] W metadata dict (linia ~234) dodaj:
   ```python
   metadata={
       "trained_at": timestamp,
       "data_points": len(X_seq),
       "loss": best_model_loss,
       "config_signature": config_signature,  # ✅ DODAJ TO
   }
   ```
6. [ ] Test manual:
   ```bash
   # Uruchom day_trading 2x z różnymi configami - powinien wykryć drift
   python main.py --strategy day_trading
   # Zmień config
   python main.py --strategy day_trading  # Powinien zalogować drift warning
   ```
7. [ ] Commit: `git commit -m "Fix BUG-3: Add config_signature to day_trading metadata"`

---

### BUG-4: Hold-out Data Leakage (short_term)
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 30 minut
**Plik:** `strategies/short_term_strategy.py`

**Problem:**
Training set (`data.iloc[:-1]`) i inference set (`data.iloc[-25:]`) mają 24/25 wierszy wspólnych - to nie jest prawdziwy out-of-sample test!

**Kroki:**
1. [ ] Otwórz `strategies/short_term_strategy.py`
2. [ ] Znajdź linię ~83-86 (train/test split)
3. [ ] Zastąp obecny split czystym podziałem:
   ```python
   # OLD (data leakage):
   # X_train = data.iloc[:-1].drop(columns=['target'])
   # y_train = data.iloc[:-1]['target']

   # NEW (clean split - ostatnie MIN_INFERENCE_ROWS poza treningiem):
   split_idx = len(data) - MIN_INFERENCE_ROWS
   X_train = data.iloc[:split_idx].drop(columns=['target'])
   y_train = data.iloc[:split_idx]['target']

   logger.info(f"Training on {len(X_train)} rows, reserving last {MIN_INFERENCE_ROWS} for inference")
   ```
4. [ ] Znajdź linię ~88-91 (inference set)
5. [ ] Zaktualizuj komentarz:
   ```python
   # Prepare inference data (truly out-of-sample - NOT in training)
   data_inference = data.iloc[-MIN_INFERENCE_ROWS:].drop(columns=['target'])
   ```
6. [ ] Dodaj validation check:
   ```python
   # Sanity check: ensure no overlap
   assert len(X_train) + len(data_inference) == len(data), "Train/inference overlap!"
   ```
7. [ ] Uruchom testy: `pytest tests/test_short_term_inference.py -v`
8. [ ] Sprawdź czy model accuracy się zmienia (expected - może być niższe, ale prawdziwe)
9. [ ] Commit: `git commit -m "Fix BUG-4: Fix hold-out data leakage in short_term"`

**Uwaga:** Po tym fixie accuracy może spaść - to jest NORMALNE! Poprzednia accuracy była zawyżona przez leakage.

---

### BUG-5: Wskaźniki Mutują DataFrame
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 1 godzina (wszystkie indicators)
**Pliki:**
- `indicators/ema.py`
- `indicators/sma.py`
- `indicators/bollinger_bands.py`
- (wszystkie indicators które mutują)

**Problem:**
Wskaźniki modyfikują `self.data` in-place, co powoduje side effects trudne do debugowania.

**Kroki dla każdego wskaźnika:**
1. [ ] Sprawdź które wskaźniki mutują DataFrame:
   ```bash
   grep -n "self.data\[" indicators/*.py
   ```
2. [ ] Dla każdego znalezionego pliku:

**Przykład dla EMA:**
3. [ ] Otwórz `indicators/ema.py`
4. [ ] Znajdź metodę `calculate()`
5. [ ] Zmień z mutation na return:
   ```python
   # PRZED:
   def calculate(self):
       self.data[f'EMA_{self.period}'] = self.data['Close'].ewm(span=self.period).mean()
       return self.data[f'EMA_{self.period}']

   # PO:
   def calculate(self) -> pd.Series:
       """Calculate EMA without mutating input data."""
       return self.data['Close'].ewm(span=self.period).mean()
   ```
6. [ ] Zaktualizuj strategie które używają tego wskaźnika:
   ```python
   # W strategiach - explicit assignment:
   data = data.copy()  # Ensure we work on copy
   data[f'EMA_{period}'] = EMA(data, period).calculate()
   ```

7. [ ] Powtórz dla:
   - [ ] `indicators/sma.py`
   - [ ] `indicators/bollinger_bands.py`
   - [ ] `indicators/stochastic.py`
   - [ ] Innych które mutują

8. [ ] Uruchom wszystkie testy indicators:
   ```bash
   pytest tests/test_indicators.py -v
   ```
9. [ ] Sprawdź że strategie dalej działają:
   ```bash
   pytest tests/test_short_term_inference.py -v
   ```
10. [ ] Commit: `git commit -m "Fix BUG-5: Remove DataFrame mutation from indicators"`

---

## 🔴 SPRINT 1: CRITICAL Issues (P0) - 1 tydzień (~4h)

### P0-1: Security - Move Emails to .env
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 15 minut
**Zależności:** Żadne

**Pliki:**
- `configs/config_day_trading.py`
- `configs/config_short_term.py`
- `configs/config_mid_term.py`
- `configs/config_long_term.py`
- `.env.example`

**Kroki:**
1. [ ] Otwórz `.env.example`
2. [ ] Dodaj:
   ```bash
   # Email notifications
   NOTIFICATION_EMAILS=user1@example.com,user2@example.com
   ```
3. [ ] Otwórz każdy config file (day_trading, short_term, mid_term, long_term)
4. [ ] Zastąp hardcoded emails:
   ```python
   # PRZED:
   "notification_email": ['mateusz.dziuk@gmail.com', 'biuro@aszenbrener.pl'],

   # PO:
   import os
   "notification_email": os.getenv("NOTIFICATION_EMAILS", "").split(","),
   ```
5. [ ] Skopiuj `.env.example` do `.env`:
   ```bash
   cp .env.example .env
   ```
6. [ ] Wpisz prawdziwe emaile do `.env`
7. [ ] Sprawdź że `.env` jest w `.gitignore`
8. [ ] Test: `python -c "from configs.config_day_trading import config; print(config['notification_email'])"`
9. [ ] **WAŻNE:** Usuń hardcoded emails z git history (jeśli były commitowane):
   ```bash
   # UWAGA: To jest destructive operation!
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch configs/config_*.py" \
     --prune-empty --tag-name-filter cat -- --all
   ```
10. [ ] Commit: `git commit -m "Security P0-1: Move notification emails to .env"`

---

### P0-2: Custom Exception Classes
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 1 godzina
**Zależności:** Żadne

**Kroki:**
1. [ ] Stwórz nowy plik `core/exceptions.py`:
   ```python
   """Custom exceptions for trading bot."""

   class TradingBotError(Exception):
       """Base exception for trading bot."""
       pass

   class InsufficientDataError(TradingBotError):
       """Raised when insufficient data for strategy execution."""
       pass

   class ModelPersistenceError(TradingBotError):
       """Raised when model save/load fails."""
       pass

   class RiskViolationError(TradingBotError):
       """Raised when trade violates risk management rules."""
       pass

   class DataValidationError(TradingBotError):
       """Raised when data validation fails."""
       pass

   class ConfigurationError(TradingBotError):
       """Raised when configuration is invalid."""
       pass
   ```

2. [ ] Zastąp broad exceptions w `utils/validators.py`:
   ```python
   from core.exceptions import DataValidationError

   # Zamiast generic Exception:
   raise DataValidationError(f"Missing required columns: {missing}")
   ```

3. [ ] Zastąp w `utils/model_persistence.py`:
   ```python
   from core.exceptions import ModelPersistenceError

   try:
       model = keras.models.load_model(model_path)
   except Exception as e:
       raise ModelPersistenceError(f"Failed to load model: {e}") from e
   ```

4. [ ] Zastąp w `strategies/*_strategy.py`:
   ```python
   from core.exceptions import InsufficientDataError

   if len(data) < MIN_ROWS:
       raise InsufficientDataError(f"Need at least {MIN_ROWS} rows, got {len(data)}")
   ```

5. [ ] Zaktualizuj główną pętlę `main.py` żeby łapać specific exceptions:
   ```python
   from core.exceptions import TradingBotError, InsufficientDataError

   try:
       strategy_instance.execute()
   except InsufficientDataError as e:
       logger.warning(f"Insufficient data: {e}")
       # Retry with backoff
   except TradingBotError as e:
       logger.error(f"Trading bot error: {e}")
       # Handle gracefully
   except Exception as e:
       logger.error(f"Unexpected error: {e}", exc_info=True)
       # Escalate
   ```

6. [ ] Dodaj testy:
   ```python
   # tests/test_exceptions.py
   from core.exceptions import InsufficientDataError

   def test_custom_exception_raised():
       with pytest.raises(InsufficientDataError):
           raise InsufficientDataError("Test")
   ```

7. [ ] Run all tests: `pytest tests/ -v`
8. [ ] Commit: `git commit -m "P0-2: Add custom exception classes"`

---

### P0-3: Add pyproject.toml + Linters
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 30 minut + formatowanie
**Zależności:** Żadne

**Kroki:**
1. [ ] Stwórz `pyproject.toml`:
   ```toml
   [project]
   name = "trading-bot"
   version = "0.1.0"
   description = "Multi-strategy trading bot with ML-powered signals"
   requires-python = ">=3.10"
   dependencies = [
       "pandas>=2.0,<3",
       "numpy>=1.26,<2",
       "scikit-learn>=1.3,<2",
       "xgboost>=1.7,<2",
       "tensorflow>=2.11,<3",
       "yfinance>=0.2,<1",
       "python-binance>=1,<2",
       "pydantic>=1.10,<3",
   ]

   [project.optional-dependencies]
   dev = [
       "pytest>=7,<9",
       "pytest-cov>=4,<6",
       "black>=23,<25",
       "ruff>=0.1,<1",
       "mypy>=1.7,<2",
   ]

   [tool.black]
   line-length = 120
   target-version = ['py310', 'py311', 'py312']
   include = '\.pyi?$'
   exclude = '''
   /(
       \.git
     | \.venv
     | venv
     | build
     | dist
     | saved_models
     | cache
   )/
   '''

   [tool.isort]
   profile = "black"
   line_length = 120

   [tool.mypy]
   python_version = "3.10"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = false
   files = ["trading-bot"]
   exclude = ["venv", ".venv", "tests"]

   [[tool.mypy.overrides]]
   module = ["yfinance.*", "binance.*", "scikeras.*"]
   ignore_missing_imports = true

   [tool.ruff]
   line-length = 120
   select = [
       "E",   # pycodestyle errors
       "W",   # pycodestyle warnings
       "F",   # pyflakes
       "I",   # isort
       "B",   # flake8-bugbear
       "C4",  # flake8-comprehensions
       "UP",  # pyupgrade
   ]
   ignore = [
       "E501",  # line too long (handled by black)
       "B008",  # function calls in argument defaults
   ]
   exclude = [".venv", "venv", "build", "dist", "saved_models", "cache"]

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   python_functions = ["test_*"]
   addopts = "-v --tb=short --cov=trading-bot --cov-report=html --cov-report=term"

   [build-system]
   requires = ["setuptools>=68", "wheel"]
   build-backend = "setuptools.build_meta"
   ```

2. [ ] Install dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. [ ] Run formatters:
   ```bash
   # Black
   black trading-bot/ --check  # Sprawdź co się zmieni
   black trading-bot/           # Formatuj

   # Isort
   isort trading-bot/ --check
   isort trading-bot/

   # Ruff
   ruff check trading-bot/      # Check issues
   ruff check trading-bot/ --fix  # Auto-fix
   ```

4. [ ] Run mypy:
   ```bash
   mypy trading-bot/
   # Spodziewaj się DUŻO błędów - to jest baseline
   ```

5. [ ] Setup pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   ```

6. [ ] Stwórz `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: https://github.com/psf/black
       rev: 23.12.1
       hooks:
         - id: black
           language_version: python3.10

     - repo: https://github.com/pycqa/isort
       rev: 5.13.2
       hooks:
         - id: isort

     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.1.9
       hooks:
         - id: ruff
           args: [--fix]
   ```

7. [ ] Install hooks:
   ```bash
   pre-commit install
   ```

8. [ ] Run hooks on all files:
   ```bash
   pre-commit run --all-files
   ```

9. [ ] Commit formatowanie:
   ```bash
   git add -A
   git commit -m "P0-3: Add pyproject.toml and run formatters (black, isort, ruff)"
   ```

---

### P0-4: Add Lock File (pip-tools)
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 1 godzina
**Zależności:** P0-3 (pyproject.toml)

**Kroki:**
1. [ ] Install pip-tools:
   ```bash
   pip install pip-tools
   ```

2. [ ] Stwórz `requirements.in` (source dependencies):
   ```txt
   # Core dependencies
   pandas>=2.0,<3
   numpy>=1.26,<2
   scikit-learn>=1.3,<2
   scikeras>=0.13,<0.14
   xgboost>=1.7,<2
   tensorflow>=2.11,<3
   yfinance>=0.2,<1
   python-binance>=1,<2
   pydantic>=1.10,<3
   ```

3. [ ] Stwórz `requirements-dev.in`:
   ```txt
   -c requirements.txt  # Constraint to main deps

   pytest>=7,<9
   pytest-cov>=4,<6
   black>=23,<25
   ruff>=0.1,<1
   mypy>=1.7,<2
   isort>=5.13,<6
   ```

4. [ ] Compile lock files:
   ```bash
   pip-compile requirements.in -o requirements.txt --resolver=backtracking
   pip-compile requirements-dev.in -o requirements-dev.txt --resolver=backtracking
   ```

5. [ ] Test installation from lock file:
   ```bash
   # Fresh venv
   python -m venv test_venv
   source test_venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Test imports
   python -c "import pandas; import tensorflow; import xgboost; print('OK')"

   deactivate
   rm -rf test_venv
   ```

6. [ ] Update README z instrukcjami:
   ```markdown
   ## Installation

   ### Production
   ```bash
   pip install -r requirements.txt
   ```

   ### Development
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

   ### Update dependencies
   ```bash
   pip-compile requirements.in -o requirements.txt --upgrade
   ```
   ```

7. [ ] Commit:
   ```bash
   git add requirements.in requirements.txt requirements-dev.in requirements-dev.txt
   git commit -m "P0-4: Add lock files with pip-compile"
   ```

---

### P0-5: Create README.md
**Status:** ❌ Do zrobienia
**Priorytet:** 🔴 CRITICAL
**Effort:** 30 minut
**Zależności:** P0-3, P0-4

**Kroki:**
1. [ ] Stwórz `README.md` w root:
   ```markdown
   # Trading Bot

   Multi-strategy cryptocurrency and stock trading bot with ML-powered signals.

   ## Features

   - 🤖 4 ML-powered strategies (LSTM, XGBoost, RandomForest)
   - 📊 Multiple data sources (Yahoo Finance, Binance)
   - 📈 Technical indicators (RSI, MACD, Bollinger Bands, ADX, Stochastic)
   - 💰 Paper trading with realistic simulation
   - 🛡️ Risk management (stop-loss, take-profit, position sizing)
   - 📧 Email notifications
   - 💾 Model persistence and caching

   ## Quick Start

   ### Prerequisites

   - Python 3.10+
   - pip

   ### Installation

   ```bash
   # Clone repository
   git clone <repo-url>
   cd trading-bot

   # Install dependencies
   pip install -r requirements.txt

   # Configure environment
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

   ### Configuration

   1. Copy `.env.example` to `.env`
   2. Set your credentials:
      - Binance API keys (optional, for crypto trading)
      - Email credentials (for notifications)
   3. Choose a strategy config in `configs/`

   ### Run

   ```bash
   # Run specific strategy
   python main.py --strategy short_term

   # Available strategies:
   # - day_trading  (LSTM, 1h candles)
   # - short_term   (XGBoost, 5-day swings)
   # - mid_term     (RandomForest, 20-day trends)
   # - long_term    (RandomForest, 50-day positions)
   ```

   ## Strategies

   | Strategy | Model | Timeframe | Target | Min Data |
   |----------|-------|-----------|--------|----------|
   | Day Trading | LSTM | 1h | Intraday | 720 rows |
   | Short-Term | XGBoost | 1d | 5 days | 80 rows |
   | Mid-Term | RandomForest | 1d | 20 days | 150 rows |
   | Long-Term | RandomForest | 1d | 50 days | 300 rows |

   ## Configuration

   Each strategy has a config file in `configs/`:

   ```python
   config = {
       "strategy": "short_term",
       "data_source": "yahoo",  # or "binance"
       "ticker": "AAPL",
       "period": "1y",
       "interval": "1d",
       "indicators": ["macd", "rsi", "bollinger_bands"],
       "risk_management": {
           "stop_loss": 0.03,      # 3%
           "take_profit": 0.05,    # 5%
           "max_position_size": 0.1,  # 10% of capital
           "trading_fee": 0.001,   # 0.1%
       },
       "notification_email": ["user@example.com"],
       "log_level": "INFO",
   }
   ```

   ## Development

   ### Install dev dependencies

   ```bash
   pip install -r requirements-dev.txt
   ```

   ### Run tests

   ```bash
   # All tests
   pytest tests/ -v

   # With coverage
   pytest tests/ --cov=trading-bot --cov-report=html

   # Specific test
   pytest tests/test_short_term_inference.py -v
   ```

   ### Code quality

   ```bash
   # Format code
   black trading-bot/
   isort trading-bot/

   # Lint
   ruff check trading-bot/

   # Type check
   mypy trading-bot/
   ```

   ## Architecture

   See [docs/architecture.md](docs/architecture.md) for detailed system design.

   ```
   trading-bot/
   ├── strategies/      # Trading strategies (base + 4 implementations)
   ├── indicators/      # Technical indicators (RSI, MACD, etc.)
   ├── utils/          # Utilities (paper trading, risk, email, etc.)
   ├── models/         # Pydantic models and ML models
   ├── configs/        # Strategy configurations
   └── tests/          # Unit and integration tests
   ```

   ## Paper Trading

   All strategies run in **paper trading mode** by default:
   - No real money involved
   - Realistic simulation with fees and slippage
   - State persisted per strategy
   - Track performance and PnL

   ## Risk Management

   Built-in risk controls:
   - **Stop-loss**: Automatic exit on losses
   - **Take-profit**: Lock in profits
   - **Position sizing**: Never risk more than configured %
   - **Fees**: Realistic trading costs

   ## Monitoring

   - **Email notifications**: Get alerts on trades
   - **Logs**: Detailed logging with configurable levels
   - **Paper trading state**: Track positions and PnL

   ## Known Issues

   See [claude-final-review.md](claude-final-review.md) for complete code review and known bugs.

   **CRITICAL:** Before production use, fix these bugs:
   1. DataValidator KeyError on missing columns
   2. Duplicate model loading in short/mid-term
   3. Config drift detection broken for day_trading
   4. Hold-out data leakage in short_term
   5. Indicators mutate DataFrame

   ## Contributing

   1. Fork the repository
   2. Create feature branch (`git checkout -b feature/amazing-feature`)
   3. Run tests and linters
   4. Commit changes (`git commit -m 'Add amazing feature'`)
   5. Push to branch (`git push origin feature/amazing-feature`)
   6. Open Pull Request

   ## License

   [Your License]

   ## Disclaimer

   **FOR EDUCATIONAL PURPOSES ONLY**

   This bot is for paper trading and learning. Do not use with real money without:
   - Thorough testing
   - Understanding the code
   - Professional financial advice
   - Proper risk management

   Trading involves substantial risk of loss.
   ```

2. [ ] Review and customize README
3. [ ] Add badges (optional):
   ```markdown
   ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
   ![Tests](https://img.shields.io/badge/tests-passing-green.svg)
   ![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
   ```

4. [ ] Commit:
   ```bash
   git add README.md
   git commit -m "P0-5: Add comprehensive README.md"
   ```

---

## 🟡 SPRINT 2: HIGH Priority (P1) - 2 tygodnie (~3 dni)

### P1-1: Template Method Pattern Refactoring
**Status:** ❌ Do zrobienia
**Priorytet:** 🟡 HIGH
**Effort:** 6 godzin
**Zależności:** BUG-2 (bo dotyczy tego samego kodu)

**Cel:** Zredukować duplikację 200-350 linii kodu w execute() każdej strategii.

**Kroki:**
1. [ ] Stwórz backup przed refaktoringiem:
   ```bash
   git checkout -b refactor-template-method
   cp -r strategies/ strategies_backup/
   ```

2. [ ] Edytuj `strategies/strategy_base.py` - dodaj template method:
   ```python
   from abc import ABC, abstractmethod
   from typing import Tuple, Optional
   import pandas as pd

   class StrategyBase(ABC):
       # ... existing __init__ ...

       def execute(self) -> None:
           """
           Template method - defines the algorithm skeleton.

           This method orchestrates the strategy execution:
           1. Prepare and validate data
           2. Add indicators
           3. Engineer features
           4. Train or load model
           5. Generate prediction
           6. Make decision
           7. Execute trade
           8. Send notification
           """
           # Step 1: Prepare data
           data = self._prepare_data()
           if not self._validate_data(data):
               return

           # Step 2: Add indicators
           data = self._add_indicators(data)

           # Step 3: Feature engineering
           features, target = self._engineer_features(data)
           if features is None:
               return

           # Step 4: Get or train model
           model = self._get_or_train_model(features, target)

           # Step 5: Generate prediction
           prediction = self._generate_prediction(model, data)

           # Step 6: Make decision
           decision, reason = self._make_decision(prediction, data)

           # Step 7: Execute trade
           trade_result = self._execute_trade(decision, data)

           # Step 8: Notify
           self._send_notification(decision, reason, trade_result)

       # Concrete methods (default implementations)
       def _prepare_data(self) -> pd.DataFrame:
           """Prepare data (can be overridden)."""
           return self.data.copy().sort_index()

       def _validate_data(self, data: pd.DataFrame) -> bool:
           """Validate data (can be overridden)."""
           if data.empty:
               self.logger.warning("Empty dataset, skipping execution.")
               return False
           if len(data) < self.MIN_ROWS:
               self.logger.warning(f"Need at least {self.MIN_ROWS} rows, got {len(data)}")
               return False
           return True

       def _execute_trade(self, decision: str, data: pd.DataFrame) -> dict:
           """Execute paper trade (can be overridden)."""
           last_close = data['Close'].iloc[-1]
           return self.order_executor.process_signal(
               self.config["ticker"],
               decision,
               last_close,
               self.risk_manager
           )

       def _send_notification(self, decision: str, reason: str, trade_result: dict) -> None:
           """Send email notification (can be overridden)."""
           # Implementation here
           pass

       # Abstract methods (must be implemented by subclasses)
       @abstractmethod
       def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
           """Add strategy-specific indicators."""
           pass

       @abstractmethod
       def _engineer_features(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
           """Engineer features and target variable."""
           pass

       @abstractmethod
       def _get_or_train_model(self, features: pd.DataFrame, target: pd.Series):
           """Get cached model or train new one."""
           pass

       @abstractmethod
       def _generate_prediction(self, model, data: pd.DataFrame) -> float:
           """Generate prediction using the model."""
           pass

       @abstractmethod
       def _make_decision(self, prediction: float, data: pd.DataFrame) -> Tuple[str, str]:
           """Make trading decision based on prediction."""
           pass
   ```

3. [ ] Refactor ShortTermStrategy jako przykład:
   ```python
   class ShortTermStrategy(StrategyBase):
       MIN_ROWS = 80

       def _add_indicators(self, data):
           # Extract from current execute(), lines ~55-72
           for indicator_name in self.config.get('indicators', []):
               # ... indicator logic
           return data

       def _engineer_features(self, data):
           # Extract from current execute(), lines ~73-99
           # Return (X_train, y_train) or (None, None) if insufficient
           pass

       def _get_or_train_model(self, features, target):
           # Extract train_or_load_pipeline logic
           return self.train_or_load_pipeline(features, target)

       def _generate_prediction(self, model, data):
           # Extract inference logic
           X_inference = ...
           return model.predict(X_inference)[0]

       def _make_decision(self, prediction, data):
           # Extract decision logic, lines ~260-276
           if prediction > 0.005:
               return "BUY", f"Predicted return: {prediction:.4f}"
           elif prediction < -0.005:
               return "SELL", f"Predicted return: {prediction:.4f}"
           else:
               return "HOLD", f"Predicted return: {prediction:.4f}"
   ```

4. [ ] Test ShortTermStrategy:
   ```bash
   pytest tests/test_short_term_inference.py -v
   ```

5. [ ] Jeśli działa, refactor remaining strategies:
   - [ ] Day Trading Strategy
   - [ ] Mid-Term Strategy
   - [ ] Long-Term Strategy

6. [ ] Run all tests:
   ```bash
   pytest tests/ -v
   ```

7. [ ] Measure LOC reduction:
   ```bash
   # Before
   wc -l strategies/*_strategy.py

   # After refactor
   wc -l strategies/*_strategy.py

   # Should see significant reduction
   ```

8. [ ] Commit:
   ```bash
   git add strategies/
   git commit -m "P1-1: Refactor strategies with Template Method Pattern"
   ```

**Oczekiwany rezultat:**
- ShortTermStrategy: 403 lines → ~150 lines
- MidTermStrategy: 221 lines → ~120 lines
- LongTermStrategy: 207 lines → ~110 lines
- DayTradingStrategy: 265 lines → ~140 lines

---

### P1-2: Measure & Increase Test Coverage
**Status:** ❌ Do zrobienia
**Priorytet:** 🟡 HIGH
**Effort:** 2-3 dni (incremental)
**Zależności:** P0-3 (pytest-cov)

**Cel:** Zmierzyć actual coverage i zwiększyć z ~20-30% do 60%+

**Kroki:**

**Faza 1: Measurement (1h)**
1. [ ] Run pytest-cov:
   ```bash
   pytest tests/ --cov=trading-bot --cov-report=html --cov-report=term
   ```
2. [ ] Otwórz `htmlcov/index.html` w przeglądarce
3. [ ] Zidentyfikuj coverage per module:
   - [ ] strategies/: ____%
   - [ ] indicators/: ____%
   - [ ] utils/: ____%
   - [ ] models/: ____%
4. [ ] Save baseline:
   ```bash
   echo "Baseline coverage: __%" > coverage_baseline.txt
   ```

**Faza 2: Critical Path Testing (1 dzień)**
5. [ ] Dodaj testy dla strategies execute():
   ```python
   # tests/test_strategies_execute.py (NEW FILE)

   def test_short_term_execute_happy_path(sample_data):
       """Test short_term execute with valid data."""
       config = load_config("short_term")
       strategy = ShortTermStrategy(config, sample_data(rows=100))

       # Should not crash
       strategy.execute()

   def test_short_term_execute_insufficient_data():
       """Test strategy handles insufficient data gracefully."""
       config = load_config("short_term")
       strategy = ShortTermStrategy(config, sample_data(rows=10))

       # Should log warning and return early
       strategy.execute()

   def test_short_term_execute_nan_indicators():
       """Test strategy handles NaN in indicators."""
       data = sample_data(rows=100)
       data.loc[50:55, 'Close'] = np.nan

       config = load_config("short_term")
       strategy = ShortTermStrategy(config, data)

       # Should handle NaN gracefully
       strategy.execute()

   # Repeat for other strategies...
   ```

6. [ ] Dodaj testy dla edge cases:
   ```python
   # tests/test_data_fetcher_edge_cases.py (NEW FILE)

   @pytest.mark.parametrize("period,interval", [
       ("1d", "1m"),   # Very short period
       ("10y", "1d"),  # Very long period
       ("1y", "1h"),   # Moderate
   ])
   def test_binance_fetch_various_periods(period, interval):
       """Test Binance fetch with various period/interval combinations."""
       # Mock or use small dataset
       data = fetch_binance_data("BTCUSDT", interval, period)
       assert not data.empty
       assert len(data) > 0

   def test_yahoo_fetch_invalid_ticker():
       """Test Yahoo Finance with invalid ticker."""
       data = fetch_yahoo_data("INVALID_TICKER_XYZ", "1y", "1d")
       assert data.empty  # Should return empty, not crash
   ```

**Faza 3: Model Persistence Testing (4h)**
7. [ ] Dodaj testy dla persistence:
   ```python
   # tests/test_model_persistence.py (ENHANCE EXISTING)

   def test_persistence_version_mismatch():
       """Test handling of version mismatch."""
       # Save with old version, load with new
       pass

   def test_persistence_corrupted_file():
       """Test handling of corrupted model file."""
       pass

   def test_persistence_concurrent_access():
       """Test concurrent read/write (if file locking added)."""
       pass
   ```

**Faza 4: Indicators Edge Cases (3h)**
8. [ ] Enhance indicator tests:
   ```python
   # tests/test_indicators.py (ENHANCE EXISTING)

   def test_rsi_with_all_same_values():
       """Test RSI when all prices are identical."""
       data = pd.DataFrame({'Close': [100] * 50}, index=pd.date_range('2024-01-01', periods=50))
       rsi = RSI(data, period=14).calculate()
       # RSI should be 50 or NaN

   def test_macd_with_insufficient_data():
       """Test MACD with fewer rows than required."""
       data = sample_data(rows=10)  # Less than MACD period
       macd = MACD(data).calculate()
       # Should handle gracefully

   def test_bollinger_bands_extreme_volatility():
       """Test Bollinger Bands during extreme price swings."""
       pass
   ```

**Faza 5: Integration Tests (1 dzień)**
9. [ ] Dodaj end-to-end tests:
   ```python
   # tests/test_integration_full_flow.py (NEW FILE)

   @pytest.mark.integration
   def test_full_flow_short_term():
       """Test complete flow: fetch → validate → execute → paper trade."""
       # 1. Fetch data
       data = fetch_data_online("yahoo", "AAPL", "1y", "1d")

       # 2. Validate
       validator = DataValidator(min_rows=80)
       result = validator.validate(data)
       assert result.is_valid

       # 3. Execute strategy
       config = load_config("short_term")
       strategy = ShortTermStrategy(config, result.data)
       strategy.execute()

       # 4. Check paper trading state
       state_file = Path("paper_trading_state_short_term.json")
       assert state_file.exists()
   ```

**Faza 6: Monitor Progress**
10. [ ] Po każdej fazie, check coverage:
    ```bash
    pytest tests/ --cov=trading-bot --cov-report=term | grep TOTAL
    ```
11. [ ] Target breakdown:
    - Baseline: ~20-30%
    - After Phase 2: ~40%
    - After Phase 3: ~50%
    - After Phase 4: ~55%
    - After Phase 5: ~60%+

12. [ ] Commit incrementally:
    ```bash
    git commit -m "P1-2: Add strategy execute tests (coverage: 40%)"
    git commit -m "P1-2: Add model persistence tests (coverage: 50%)"
    git commit -m "P1-2: Add indicator edge case tests (coverage: 55%)"
    git commit -m "P1-2: Add integration tests (coverage: 60%+)"
    ```

---

### P1-3: Add Type Hints to Public Functions
**Status:** ❌ Do zrobienia
**Priorytet:** 🟡 HIGH
**Effort:** 4 godzin (incremental)
**Zależności:** P0-3 (mypy)

**Kroki:**
1. [ ] Run mypy baseline:
   ```bash
   mypy trading-bot/ > mypy_baseline.txt
   # Zapisz liczbe błędów: ___
   ```

2. [ ] Dodaj type hints do main.py:
   ```python
   from typing import Optional

   def run_trading_bot(strategy: str) -> None:
       """Run trading bot continuously with specified strategy."""
       ...

   def _warn_on_config_drift(strategy_name: str, config: dict) -> None:
       """Compare runtime config hash with persisted metadata."""
       ...
   ```

3. [ ] Dodaj type hints do strategy_manager.py:
   ```python
   import pandas as pd
   from strategies.strategy_base import StrategyBase

   def select_strategy(config: dict, data: pd.DataFrame) -> StrategyBase:
       """Select the appropriate strategy class based on config."""
       ...
   ```

4. [ ] Dodaj type hints do data_fetcher.py:
   ```python
   import pandas as pd

   def fetch_yahoo_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
       """Fetch data from Yahoo Finance."""
       ...

   def fetch_binance_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
       """Fetch data from Binance."""
       ...

   def fetch_data_online(
       source: str = 'yahoo',
       ticker: str = 'BTCUSDT',
       period: str = '1y',
       interval: str = '1d'
   ) -> pd.DataFrame:
       """Fetch data from specified source."""
       ...
   ```

5. [ ] Dodaj type hints do strategies (if refactored with Template Method):
   ```python
   from typing import Tuple, Optional
   import pandas as pd

   class ShortTermStrategy(StrategyBase):
       def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
           ...

       def _engineer_features(
           self,
           data: pd.DataFrame
       ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
           ...

       def _make_decision(
           self,
           prediction: float,
           data: pd.DataFrame
       ) -> Tuple[str, str]:
           ...
   ```

6. [ ] Run mypy after each module:
   ```bash
   mypy trading-bot/main.py
   mypy trading-bot/strategy_manager.py
   mypy trading-bot/data_fetcher.py
   # etc.
   ```

7. [ ] Target modules priority:
   - [x] main.py
   - [x] strategy_manager.py
   - [x] data_fetcher.py
   - [ ] strategies/strategy_base.py
   - [ ] strategies/*_strategy.py
   - [ ] indicators/indicator_base.py
   - [ ] utils/validators.py
   - [ ] utils/risk_management.py

8. [ ] Create type stubs for external libraries (if needed):
   ```python
   # typings/yfinance.pyi
   import pandas as pd

   def download(ticker: str, period: str, interval: str) -> pd.DataFrame: ...
   ```

9. [ ] Final mypy run:
   ```bash
   mypy trading-bot/ > mypy_after.txt
   # Porównaj z baseline - powinno być mniej błędów
   ```

10. [ ] Commit:
    ```bash
    git add -A
    git commit -m "P1-3: Add type hints to public functions (mypy errors: X→Y)"
    ```

---

### P1-4: Add Docstrings to Public Functions
**Status:** ❌ Do zrobienia
**Priorytet:** 🟡 HIGH
**Effort:** 3 godziny (incremental)
**Zależności:** Żadne

**Cel:** Google-style docstrings dla wszystkich public functions

**Template:**
```python
def function_name(arg1: str, arg2: int) -> bool:
    """
    Short one-line summary.

    Optional longer description explaining what the function does,
    its behavior, and any important details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid
        IOError: When file operation fails

    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

**Kroki:**
1. [ ] Dodaj docstrings do main.py:
   ```python
   def run_trading_bot(strategy: str) -> None:
       """
       Run trading bot continuously with specified strategy.

       This function:
       1. Loads and validates configuration
       2. Enters infinite loop to fetch data and execute strategy
       3. Implements exponential backoff on errors
       4. Respects strategy-specific sleep durations

       Args:
           strategy: Strategy name ('day_trading', 'short_term', 'mid_term', 'long_term')

       Raises:
           ValueError: If invalid strategy or configuration

       Example:
           >>> run_trading_bot('short_term')
       """
       ...
   ```

2. [ ] Dodaj do strategy_manager.py:
   ```python
   def select_strategy(config: dict, data: pd.DataFrame) -> StrategyBase:
       """
       Select and instantiate strategy based on configuration.

       Uses factory pattern to create strategy instance. Strategies
       are lazy-loaded to improve startup performance.

       Args:
           config: Strategy configuration dict with 'strategy' key
           data: Market data DataFrame with OHLCV columns

       Returns:
           Instance of requested strategy (subclass of StrategyBase)

       Raises:
           ValueError: If strategy type is unsupported

       Example:
           >>> config = {'strategy': 'short_term', ...}
           >>> data = pd.DataFrame(...)
           >>> strategy = select_strategy(config, data)
           >>> strategy.execute()
       """
       ...
   ```

3. [ ] Dodaj do data_fetcher.py functions
4. [ ] Dodaj do wszystkich strategy classes
5. [ ] Dodaj do indicator classes
6. [ ] Dodaj do utils functions

7. [ ] Generate docs (optional):
   ```bash
   pip install pdoc3
   pdoc --html trading-bot -o docs/api
   ```

8. [ ] Commit:
   ```bash
   git add -A
   git commit -m "P1-4: Add Google-style docstrings to public functions"
   ```

---

## 🟢 SPRINT 3: MODERATE Priority (P2) - 2 tygodnie (~3 dni)

*(Reszta tasków P2-1 przez P2-10 - szczegóły w kolejnej sekcji...)*

---

## 📊 Progress Tracking

### Coverage Metrics
- [ ] Baseline measured: ____%
- [ ] Current: ____%
- [ ] Target: 60%+

### Type Hints Metrics
- [ ] Baseline mypy errors: ___
- [ ] Current errors: ___
- [ ] Target: <50 errors

### Completed Tasks
- Sprint 0 (BUGS): 0/5
- Sprint 1 (P0): 0/5
- Sprint 2 (P1): 0/4
- Sprint 3 (P2): 0/10

**Total: 0/24 (0%)**

---

## 🎯 Next Steps

1. **START HERE:** Sprint 0 - Fix 5 CRITICAL BUGS (~2h)
2. Then: Sprint 1 - P0 issues (~4h)
3. Then: Sprint 2 - P1 improvements (~3 days)
4. Finally: Sprint 3 - P2 enhancements (~3 days)

**Estimated total time:** ~2 tygodnie full-time lub ~4 tygodnie part-time

---

**Generated from:** claude-final-review.md
**Last updated:** 2025-12-04
