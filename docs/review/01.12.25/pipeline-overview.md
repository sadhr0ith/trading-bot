# Pipeline Overview

## Training
- Strategies build sklearn/Keras pipelines with feature transformers (lags, returns, rolling stats, indicators, calendar) plus scaler/model.
- TimeSeriesSplit walk-forward CV is used for tabular models; LSTM uses a small grid with TimeSeriesSplit on sequences.
- Models persist per strategy with metadata (features, config signatures, cv metrics, version, timestamp).

## Prediction
- Inference always loads the latest persisted artifact; no manual scaling/FE outside the pipeline.
- LSTM inference uses the saved scalers and last window only.

## Naming conventions
- Indicators: `MACD`, `Signal`, `MACD_Histogram`, `RSI`, `BB_Middle`, `BB_Upper`, `BB_Lower`, `BB_Width`, `ADX`, `Plus_DI`, `Minus_DI`.
- Lags are suffixed `_lag_{n}`; calendar features: `day_of_week`, `month`, `is_month_start`, `is_month_end`.

## Persistence
- Artifacts stored under `saved_models/<strategy>/<version>.*` with JSON metadata.
- Config signatures guard reuse (feature set, model params/search space, data index).
