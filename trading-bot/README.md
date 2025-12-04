# Trading Bot

Multi-strategy trading bot (crypto/stocks) with ML pipelines, technical indicators, paper-trading executor i prostą obsługą ryzyka.

## Wymagania
- Python 3.12 (zalecany)
- Zależności z `requirements.txt` (produkcyjne) oraz `requirements-dev.txt` (dev/test)

## Instalacja
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# dla pracy developerskiej
pip install -r requirements-dev.txt
```

## Konfiguracja
1) Skopiuj `.env.example` → `.env` i uzupełnij:
   - `BINANCE_API_KEY` / `BINANCE_API_SECRET` (opcjonalnie dla źródła binance)
   - `GMAIL_SENDER_EMAIL` / `GMAIL_APP_PASSWORD` (dla powiadomień e-mail)
   - `NOTIFICATION_EMAILS` (lista odbiorców, rozdzielona przecinkami)
   - `TRADING_BOT_CACHE_TTL_SECONDS` (opcjonalny cache fetcha)
2) Wybierz/edytuj konfigurację strategii w `configs/config_<strategy>.py` (ticker, okres, interwał, wskaźniki, progi ryzyka).

## Uruchomienie
```bash
python main.py --strategy short_term
# dostępne: day_trading | short_term | mid_term | long_term
```

## Funkcje
- 4 strategie:
  - `day_trading` (LSTM, 1h)
  - `short_term` (XGBoost, 5-dniowy horyzont)
  - `mid_term` (RandomForest, 20 dni)
  - `long_term` (RandomForest, 50 dni)
- Wskaźniki: RSI, MACD, SMA/EMA, Bollinger Bands, ADX, Stochastic.
- Paper trading z izolowanym stanem per strategia, obsługa SL/TP/fee.
- Persistencja modeli (sklearn/keras) i detekcja driftu konfigu.
- Walidacja danych (Pydantic + sanity checks OHLCV).

## Testy i jakość
```bash
pytest tests/ -v
ruff check trading-bot/
black trading-bot/
isort trading-bot/
mypy trading-bot/
```

## Struktura
```
trading-bot/
├── configs/           # konfiguracje strategii
├── strategies/        # implementacje strategii + StrategyBase
├── indicators/        # wskaźniki techniczne
├── utils/             # walidatory, logowanie, cache, persistence, paper trading
├── models/            # modele Pydantic + konstruktor LSTM
├── tests/             # testy jednostkowe/integracyjne
└── saved_models/      # artefakty modeli (lokalnie, ignorowane w VCS)
```

## Uwagi operacyjne
- Bot domyślnie działa w trybie paper trading; brak realnej egzekucji.
- Dane Binance pobierane są stronicowane; warto ustawić cache TTL dla lżejszych startów.
- Retraining jest kosztowny (szczególnie LSTM); w środowisku prod rozważ rozdzielenie jobów trening/inferencja.
