# Pydantic Config Schema

**Data:** 2025-12-03  
**Zakres:** JSON Schema i przykłady konfiguracji po migracji na Pydantic

## Strategie (`StrategyConfig`)
- `strategy` (enum): `day_trading | short_term | mid_term | long_term`
- `data_source` (enum): `yahoo | binance`
- `ticker` (string): symbol instrumentu
- `period` (string): okres zgodny z `parse_period_to_timedelta` (np. `1d`, `1w`, `6M`, `1y`)
- `interval` (enum zależny od źródła):  
  - Binance: `1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w`  
  - Yahoo: `1d, 1wk, 1mo, 1h, 90m`
- `indicators` (set[str]): podzbiór `rsi, macd, sma, ema, stochastic, bollinger_bands, adx`
- `use_indicators` (bool, domyślnie `true`): gdy `false` → `indicators` musi być puste
- `log_level` (str|int, opcjonalne): poziom logowania
- `notification_email` (EmailStr | list[EmailStr], opcjonalne)
- `seed` (int, opcjonalne)
- `risk_management` (RiskConfig): osadzone ustawienia ryzyka
- `min_rows` (int, opcjonalne): minimalna liczba wierszy danych

## Ryzyko (`RiskConfig`)
- `stop_loss` (float 0..1, opcjonalnie)
- `take_profit` (float 0..1, opcjonalnie)
- `max_position_size` (float 0..1, domyślnie `0.1`)
- `trading_fee` (float 0..1, domyślnie `0.001`)

## Ustawienia środowiskowe (BaseSettings)
- `BinanceSettings`: `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `EmailSettings`: `GMAIL_SENDER_EMAIL`, `GMAIL_APP_PASSWORD`
- `CacheSettings`: `TRADING_BOT_CACHE_TTL_SECONDS` (int ≥ 0)
- `LoggingSettings`: `TRADING_BOT_LOG_LEVEL` (opcjonalny)

## Przykładowa konfiguracja (JSON)
```json
{
  "strategy": "short_term",
  "data_source": "binance",
  "ticker": "BTCUSDT",
  "period": "6M",
  "interval": "1h",
  "indicators": ["macd", "rsi"],
  "use_indicators": true,
  "notification_email": ["alerts@example.com"],
  "log_level": "DEBUG",
  "risk_management": {
    "stop_loss": 0.03,
    "take_profit": 0.05,
    "max_position_size": 0.1,
    "trading_fee": 0.001
  },
  "seed": 42
}
```

## JSON Schema (skrót)
- Strategia: `oneOf` enum strategii + walidacja interwału zależna od `data_source`
- Ryzyko: każdy parametr z ograniczeniem `minimum=0`, `maximum=1`
- ENV: BaseSettings z aliasami env (`env` w polach), bez dodatkowych kluczy (`extra="ignore"`)
