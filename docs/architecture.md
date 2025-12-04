# Trading Bot – Architecture Overview

**Cel dokumentu:** Zrozumienie architektury projektu, głównych wzorców projektowych i flow danych.
**Dla kogo:** Developerzy pracujący z projektem, kontrybutorzy, code reviewerzy.
**Data:** 2025-12-02

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         MAIN.PY                             │
│              (Orchestrator + Event Loop)                    │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌─────────────┐    ┌──────────────┐   ┌─────────────┐
│ Config      │    │ Data Fetcher │   │ Validators  │
│ Handler     │    │ (Yahoo/      │   │ (Config +   │
│             │    │  Binance)    │   │  Data)      │
└─────────────┘    └──────────────┘   └─────────────┘
        │                  │
        └────────┬─────────┘
                 ▼
        ┌─────────────────┐
        │ Strategy Manager│
        │  (Factory)      │
        └─────────────────┘
                 │
     ┌───────────┴───────────┬───────────┬─────────────┐
     ▼                       ▼           ▼             ▼
┌──────────┐       ┌──────────────┐  ┌─────────┐  ┌──────────┐
│Day Trading│       │ Short-Term   │  │Mid-Term │  │Long-Term │
│(LSTM)    │       │ (XGBoost)    │  │(RandomF)│  │(RandomF) │
└──────────┘       └──────────────┘  └─────────┘  └──────────┘
     │                    │               │            │
     └────────────────────┴───────────────┴────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    ┌──────────────┐           ┌───────────────┐
    │ Paper Trading│           │ Risk Manager  │
    │  Executor    │◄──────────│               │
    └──────────────┘           └───────────────┘
            │
            ▼
    ┌──────────────┐
    │ Notifications│
    │  (Email)     │
    └──────────────┘
```

---

## 📁 Struktura Projektu

```
trading-bot/
├── main.py                      # Entry point, main loop
├── config_handler.py            # Config loading + sleep intervals
├── data_fetcher.py              # Yahoo Finance & Binance API
├── strategy_manager.py          # Strategy selection (factory pattern)
│
├── configs/                     # Strategy configurations
│   ├── config_day_trading.py
│   ├── config_short_term.py
│   ├── config_mid_term.py
│   └── config_long_term.py
│
├── strategies/                  # Trading strategies (Strategy Pattern)
│   ├── strategy_base.py         # ABC base class
│   ├── day_trading_strategy.py  # LSTM dla intraday
│   ├── short_term_strategy.py   # XGBoost dla 5-day horizon
│   ├── mid_term_strategy.py     # RandomForest dla 20-day
│   └── long_term_strategy.py    # RandomForest dla 50-day
│
├── indicators/                  # Technical indicators
│   ├── indicator_base.py        # ABC base dla wskaźników
│   ├── macd.py, rsi.py, adx.py, etc.
│
├── models/                      # [UNUSED] Model wrappers
│   └── (dead code - do usunięcia)
│
├── utils/                       # Utilities & helpers
│   ├── transformers.py          # Sklearn feature engineering
│   ├── model_persistence.py     # Versioned model saving/loading
│   ├── strategy_helpers.py      # train_or_load_pipeline()
│   ├── paper_trading.py         # Simulated trade execution
│   ├── risk_management.py       # Position sizing, stop-loss/TP
│   ├── validators.py            # Config & data validation
│   ├── logger.py                # Centralized logging setup
│   ├── email_notifications.py   # SMTP email alerts
│   └── seeding.py               # Reproducibility (random seeds)
│
└── tests/                       # Unit & integration tests
    ├── test_data_fetcher.py
    ├── test_transformers.py
    ├── test_risk_paper_integration.py
    └── ...
```

---

## 🔄 Data Flow & Execution Lifecycle

### 1. **Initialization (Start)**
```python
main.py --strategy short_term
  ↓
config_handler.load_config("short_term")
  ↓
ConfigValidator.validate(config)  # Sprawdza required keys
  ↓
setup_logger(level=config["log_level"])
```

### 2. **Main Loop (Infinite)**
```python
while True:
    # FETCH
    data = fetch_data_online(
        source=config["data_source"],  # "yahoo" lub "binance"
        ticker=config["ticker"],        # "BTCUSDT"
        period=config["period"],        # "6mo"
        interval=config["interval"]     # "1d"
    )

    # VALIDATE
    validation = DataValidator(min_rows=50).validate(data)
    if not validation.is_valid:
        sleep(backoff_seconds)
        continue

    # EXECUTE STRATEGY
    strategy = select_strategy(config, data)  # Factory
    strategy.execute()  # Core logic

    # SLEEP
    sleep_duration = get_sleep_duration(strategy)
    time.sleep(sleep_duration)
```

### 3. **Strategy Execution Flow**
```python
strategy.execute():
    # 1. INDICATORS
    data = add_indicators(data)  # MACD, RSI, ADX, etc.

    # 2. FEATURE ENGINEERING
    data = create_features(data)  # Lags, returns, rolling stats, calendar

    # 3. TRAIN/LOAD MODEL
    pipeline, cv_score = train_or_load_pipeline(
        key="short_term",
        pipeline_factory=lambda: Pipeline([...]),
        X=X_train,
        y=y_train,
        persistence=ModelPersistence()
    )
    # Cache logic: sprawdza trained_until, feature_columns, config_signature

    # 4. PREDICT
    predicted_return = pipeline.predict(latest_data)

    # 5. DECISION LOGIC
    decision = "BUY" | "SELL" | "HOLD"

    # 6. PAPER TRADE
    trade_summary = order_executor.process_signal(
        asset, decision, current_price, risk_manager
    )

    # 7. NOTIFY
    send_email(subject, body, recipients)
```

---

## 🎯 Design Patterns & Best Practices

### **1. Strategy Pattern**
**Gdzie:** `strategies/strategy_base.py` + konkretne strategie
**Dlaczego:** Pozwala na łatwe dodawanie nowych strategii bez modyfikacji istniejącego kodu (Open/Closed Principle).

```python
class StrategyBase(ABC):
    @abstractmethod
    def execute(self):
        pass

class ShortTermStrategy(StrategyBase):
    def execute(self):
        # Implementation specific to short-term trading
```

**Mentor Tip:** Każda nowa strategia dziedziczy z `StrategyBase`, dostaje automatycznie:
- `self.logger` – centralized logging
- `self.risk_manager` – position sizing, stop-loss
- `self.order_executor` – paper trading
- `self.seed` – reproducibility

---

### **2. Factory Pattern**
**Gdzie:** `strategy_manager.py:select_strategy()`
**Dlaczego:** Centralizuje tworzenie obiektów, ułatwia testowanie i modyfikację.

```python
def select_strategy(config, data):
    if config["strategy"] == "short_term":
        return ShortTermStrategy(config, data)
    elif config["strategy"] == "long_term":
        return LongTermStrategy(config, data)
    # ...
```

**Mentor Tip:** Unikamy `if strategy == "x"` w wielu miejscach. Jedna funkcja = jedno miejsce zmian.

---

### **3. Pipeline Pattern (Sklearn)**
**Gdzie:** `strategies/short_term_strategy.py`, `utils/transformers.py`
**Dlaczego:** Feature engineering + model jako jeden atomic unit. Testowanie, persistence, deploy stają się proste.

```python
pipeline = Pipeline([
    ('lag_features', LagFeatureTransformer(columns=['Close'], lags=[1,3,5,10])),
    ('return_features', ReturnFeatureTransformer(periods=[1,3,5,10])),
    ('rolling_stats', RollingStatsTransformer(windows=[5,10])),
    ('calendar_features', CalendarFeatureTransformer()),
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(...))
])

pipeline.fit(X_train, y_train)
pipeline.predict(X_new)  # All transforms + predict in one call
```

**Mentor Tip:**
- Transformery muszą być **stateless** (nie przechowywać danych między wywołaniami)
- `fit()` musi być idempotentne
- `transform()` nie może modyfikować stanu

---

### **4. Persistence with Versioning**
**Gdzie:** `utils/model_persistence.py`, `utils/strategy_helpers.py`
**Dlaczego:** Nie retrenujemy modelu przy każdym uruchomieniu, jeśli dane się nie zmieniły.

**Cache Key Components:**
1. `trained_until` – ostatni timestamp w danych
2. `feature_columns` – lista features (zmiana = retrain)
3. `config_signature` – MD5 hash hyperparametrów (zmiana = retrain)

```python
artifact = persistence.load("short_term")
if artifact:
    meta = artifact["metadata"]
    if (meta["trained_until"] == latest_timestamp and
        meta["feature_columns"] == current_features and
        meta["config_signature"] == current_config_hash):
        return artifact["model"]  # Use cached
# Else: retrain
```

**Mentor Tip:** Bez tego każde uruchomienie = 5-10 min treningu. Z cache = instant inference.

---

### **5. Time Series Validation**
**Gdzie:** Wszystkie strategie używają `TimeSeriesSplit`
**Dlaczego:** Standardowy cross-validation (KFold) jest **błędny** dla szeregów czasowych - leakuje przyszłość do przeszłości.

```python
# ❌ WRONG - shuffles data
from sklearn.model_selection import KFold

# ✅ CORRECT - respects time order
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

**Time Series Split Visual:**
```
Train: [0---1---2---3] | Test: [4]
Train: [0---1---2---3---4] | Test: [5]
Train: [0---1---2---3---4---5] | Test: [6]
...
```

**Mentor Tip:** Zawsze trenuj na przeszłości, testuj na przyszłości. Nigdy odwrotnie.

---

## 🧠 Machine Learning Components

### **Feature Engineering Transformers** (`utils/transformers.py`)

Wszystkie dziedziczą z `BaseEstimator` + `TransformerMixin` (sklearn API):

| Transformer | Co robi | Przykład |
|------------|---------|----------|
| `LagFeatureTransformer` | Tworzy shifted features | `Close_lag_1`, `Close_lag_5` |
| `ReturnFeatureTransformer` | Percent change | `Return_lag_1 = Close.pct_change(1)` |
| `RollingStatsTransformer` | Moving averages, volatility | `SMA_10`, `Volatility_5` |
| `CalendarFeatureTransformer` | Time-based features | `day_of_week`, `month`, `is_month_end` |
| `IndicatorLagTransformer` | Lags dla MACD, RSI, etc. | `MACD_lag_1`, `RSI_lag_1` |
| `FeatureSelector` | Wybiera kolumny + handle NaN | Only selected features |

**Kluczowa właściwość:** `shift()` i `rolling()` **NIE** leakują przyszłości:
```python
df['Close_lag_1'] = df['Close'].shift(1)  # Row i uses row i-1
df['SMA_5'] = df['Close'].rolling(5).mean()  # Row i uses rows i-4 to i
```

---

### **Model Strategies**

| Strategy | Model | Horizon | Key Features | Sleep Interval |
|----------|-------|---------|--------------|----------------|
| **Day Trading** | LSTM (Keras) | Next close price | Close, RSI, Volume, rolling volatility (seq=60) | 1 hour |
| **Short-Term** | XGBoost | 5-day return | Lags, returns, rolling, MACD, RSI, ADX (+calendar) | 1 day |
| **Mid-Term** | RandomForest | 20-day return | Longer lags, volatility, drawdown, MACD/BB (deduped) | 7 days |
| **Long-Term** | RandomForest | 50-day return | Very long lags (250), volatility/drawdown, SMA(200), EMA(50), MACD (deduped) | 30 days |

**Hyperparameter Tuning:**
- Day Trading: Manual grid (3 configs)
- Short-Term: `RandomizedSearchCV` (8 iterations)
- Mid-Term: `HalvingRandomSearchCV` (adaptive)
- Long-Term: No tuning (fixed config)

**Baseline Comparison:** Wszystkie strategie porównują się z **ElasticNet** baseline; day-trading loguje także naiwny baseline (ostatni close).

**Model rationale (Task #14.8):**
- LSTM (day) łapie krótkoterminowe sekwencje i nieliniowości; baseline to “ostatni close” (kontrola sanity).
- XGBoost (short) radzi sobie z heterogenicznymi cechami i interakcjami; mały RandomizedSearch przy małych próbkach ogranicza overfit i czas.
- RandomForest (mid/long) zapewnia odporność na szum; cechy uproszczone przez usunięcie duplikatów i korelacji ~1.0, co skraca inference/training.

**Inference stability (Task #14.7):** short-term loguje statystyki cech inference (NaN/min/median/max); day-trading loguje statystyki okna train/inference.

---

## 💰 Paper Trading & Risk Management

### **Paper Trading Executor** (`utils/paper_trading.py`)

**State Management:**
```json
{
  "balance": 100000.0,
  "positions": {
    "BTCUSDT": {
      "entry_price": 50000,
      "size": 0.5,
      "stop_loss": 0.03,
      "take_profit": 0.05,
      "opened_at": "2025-12-01T10:00:00"
    }
  },
  "history": [...]
}
```

**Process Flow (status: fixed):**
```python
process_signal(asset, decision, price, risk_manager):
    if decision == "BUY":
        size = risk_manager.calculate_position_size(balance, price)
        notional = price * size
        fee = notional * trading_fee
        balance -= notional + fee
        positions[asset] = {..., "entry_notional": notional}

    elif decision == "SELL":
        notional = price * size
        fee = notional * trading_fee
        balance += notional - fee
        pnl = (price - entry_price) * size - entry_fee - fee
        positions.pop(asset)

    elif decision == "HOLD":
        if risk_manager.should_exit(position, price):
            close_position()
```

**State file naming:** domyślnie `paper_trading_state_{strategy_name}.json` (per strategia); loader ostrzega, gdy strategy_name w pliku nie pasuje do bieżącej strategii.

### Troubleshooting (common)
- Brak danych / zbyt mało wierszy: sprawdź logi `DataValidator` (min_rows).  
- Env do e-maili/binance: ustaw `GMAIL_SENDER_EMAIL`, `GMAIL_APP_PASSWORD`, `BINANCE_API_KEY`, `BINANCE_API_SECRET`.  
- Cache nieaktualny: ustaw `TRADING_BOT_CACHE_TTL_SECONDS=0` lub usuń pliki w `cache/`.  
- State leakage: upewnij się, że pliki `paper_trading_state_<strategy>.json` są izolowane per strategia.

---

### **Risk Manager** (`utils/risk_management.py`)

**Configuration:**
```python
risk_config = {
    "stop_loss": 0.03,        # 3% stop loss
    "take_profit": 0.05,      # 5% take profit
    "max_position_size": 0.1, # 10% of balance
    "trading_fee": 0.001      # 0.1% fee
}
```
Dozwolone klucze: `stop_loss`, `take_profit`, `max_position_size`, `trading_fee`. Inne klucze są logowane jako ostrzeżenie i ignorowane.

**Position Sizing:**
```python
def calculate_position_size(balance, price):
    return (balance * max_position_size) / price

# Example: balance=100k, max=0.1, price=50k
# → size = (100k * 0.1) / 50k = 0.2 BTC
```

**Exit Conditions:**
```python
def evaluate_exit(position, current_price):
    stop_price = entry_price * (1 - stop_loss)
    tp_price = entry_price * (1 + take_profit)

    if current_price <= stop_price:
        return True  # Stop-loss triggered
    if current_price >= tp_price:
        return True  # Take-profit triggered
    return False
```

---

## 📊 Technical Indicators

**Base Class:** `indicators/indicator_base.py`

Wszystkie dziedziczą z `IndicatorBase`:
```python
class IndicatorBase(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def calculate(self) -> pd.Series | pd.DataFrame:
        pass
```

**Implemented Indicators:**

| Indicator | Parameters | Output Columns |
|-----------|-----------|----------------|
| MACD | fast=12, slow=26, signal=9 | MACD, Signal, MACD_Histogram |
| RSI | period=14 | RSI |
| ADX | period=14 | ADX, Plus_DI, Minus_DI |
| Bollinger | period=20, std=2 | BB_Upper, BB_Middle, BB_Lower, BB_Width |
| SMA | window | SMA |
| EMA | span | EMA |
| Stochastic | k_period=14, d_period=3 | %K, %D |

**Usage:**
```python
macd = MACD(data)
data = data.join(macd.calculate())  # Adds 3 columns
```

---

## 🔍 Data Sources

### **Yahoo Finance** (`yfinance`)
```python
data = yf.download(
    ticker="AAPL",
    period="1y",      # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    interval="1d"     # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
)
```
**Pros:** Free, no auth, easy
**Cons:** Rate limits, może czasem failować, tylko publiczne markets

### **Binance** (`python-binance`)
```python
client = Client(api_key, api_secret)
klines = client.get_klines(
    symbol="BTCUSDT",
    interval="1d",    # 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M
    startTime=...,
    limit=1000        # max per request
)
```
**Pros:** Crypto, reliable, real-time
**Cons:** Wymaga API credentials, rate limits

**Index & quality:** `fetch_binance_data` ustawia `DatetimeIndex` na open time, waliduje monotoniczność i zgłasza luki w timestampach; loguje także podstawowe metryki jakości (NaN, liczba wierszy).  
**Caching:** Optional cache TTL (env `TRADING_BOT_CACHE_TTL_SECONDS`) zapisuje dane do `cache/*.pkl` per source/ticker/period/interval.

### **Period string convention** (Task #2)
- `m` = minutes, `h` = hours, `d` = days, `w` = weeks, `M` = months (~30d), `y` = years (~365d)
- Przykłady: `"1d"`, `"6M"`, `"1y"`, `"30m"`
- Parser: `utils.time_utils.parse_period_to_timedelta`

---

## 🧪 Testing Strategy

### **Test Organization**
```
tests/
├── test_data_fetcher.py       # Mock API calls
├── test_indicators.py         # Expected indicator values
├── test_transformers.py       # Feature engineering correctness
├── test_validators.py         # Config/data validation
├── test_risk_paper_integration.py  # Risk + paper trading together
├── test_pipelines_fit_predict.py   # End-to-end ML pipeline
├── test_parse_period_extended.py   # Extended period parsing
└── test_integration_optional.py    # Lightweight integration (pipeline save/load, stop-loss HOLD, zero balance)
```

### **Testing Principles**
1. **Unit Tests:** Każdy transformer, każdy wskaźnik independently
2. **Integration Tests:** Full pipeline (data → features → model → prediction)
3. **Mock External APIs:** Nie rób real API calls w testach
4. **Time Series Fixtures:** Używaj synthetic data ze znanymi patterns

**Przykład:**
```python
def test_lag_transformer():
    df = pd.DataFrame({'Close': [10, 20, 30, 40]})
    transformer = LagFeatureTransformer(columns=['Close'], lags=[1])
    result = transformer.fit_transform(df)

    assert result['Close_lag_1'].tolist() == [np.nan, 10, 20, 30]
```

---

## ⚠️ Known Issues & Tech Debt

**Critical (P0):**
1. 🔴 **Paper trading cash accounting** - BUY/SELL nie zarządza notional poprawnie
2. 🔴 **Binance data index** - numeric zamiast DateTimeIndex
3. 🔴 **Short-term inference** - single row → NaN features
4. 🔴 **Period parsing** - brak `utils.time_utils`, tests failują
5. 🔴 **Risk config keys** - mismatch między tests a code
6. 🔴 **Hardcoded email** - security risk

**Secondary (P1):**
7. 🟡 Shared paper trading state → position leakage
8. 🟡 Duplicate features (momentum = return) w mid/long-term
9. 🟡 Dead code (`models/*`, `backtesting.py`)
10. 🟡 Logging inconsistency (data_fetcher)

**Szczegóły:** Zobacz `docs/tasklist-bugfix-after-refactor.md`

---

## 🚀 Adding a New Strategy

**Template:**
```python
# strategies/my_new_strategy.py
from .strategy_base import StrategyBase

class MyNewStrategy(StrategyBase):
    def execute(self):
        # 1. Get data
        data = self.data.copy()

        # 2. Add indicators
        data = self._add_indicators(data)

        # 3. Feature engineering
        data = self._create_features(data)

        # 4. Train/load model
        pipeline = self._train_or_load_model(data)

        # 5. Predict
        prediction = pipeline.predict(latest_data)

        # 6. Decision logic
        decision = self._make_decision(prediction)

        # 7. Execute trade
        self.order_executor.process_signal(
            self.config["ticker"],
            decision,
            current_price,
            self.risk_manager
        )

        # 8. Notify
        send_email(...)
```

**Register w strategy_manager.py:**
```python
def select_strategy(config, data):
    if config["strategy"] == "my_new":
        return MyNewStrategy(config, data)
    # ...
```

**Create config:**
```python
# configs/config_my_new.py
CONFIG = {
    "strategy": "my_new",
    "ticker": "ETHUSDT",
    "period": "1y",
    "interval": "1h",
    "indicators": ["macd", "rsi"],
    "risk_management": {...},
    "seed": 42
}
```

---

## 📚 Further Reading

**Wewnętrzne:**
- `docs/tasklist-bugfix-after-refactor.md` – szczegółowa lista bugów + fixes
- `merged-after-refactor-overview.md` – podsumowanie refactoringu
- `gpt-after-refactor-review.md` – code review findings

**Zewnętrzne:**
- [Scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- [TimeSeriesSplit CV](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [Keras/TensorFlow](https://keras.io/)
- [Python-Binance](https://python-binance.readthedocs.io/)

---

**Questions? Issues?** Zobacz tasklist lub otwórz issue.
