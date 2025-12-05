from datetime import datetime, timezone

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from indicators.rsi import RSI
from models.lstm_model import create_lstm_model
from strategies.strategy_base import StrategyBase
from utils.email_notifications import send_email
from utils.model_persistence import ModelPersistence
from utils.strategy_helpers import _build_config_signature


class DayTradingStrategy(StrategyBase):
    SEQ_LEN = 60
    MIN_ROWS = 60
    MIN_RETRAIN_SECONDS = 3600

    @staticmethod
    def _create_sequences(values: np.ndarray, targets: np.ndarray, seq_len: int):
        sequences, labels = [], []
        for i in range(len(values) - seq_len):
            sequences.append(values[i : i + seq_len])
            labels.append(targets[i + seq_len])
        return np.array(sequences), np.array(labels)

    @staticmethod
    def _latest_index(data):
        return str(data.index.max()) if hasattr(data, "index") else None

    @staticmethod
    def _scale_split(scaler_X, scaler_y, X_raw, y_raw, feature_count):
        X_scaled = scaler_X.transform(X_raw.reshape(-1, feature_count)).reshape(X_raw.shape)
        y_scaled = scaler_y.transform(y_raw.reshape(-1, 1))
        return X_scaled, y_scaled

    def _build_tuner_grid(self):
        return [
            {"units": 32, "learning_rate": 0.001, "epochs": 18, "batch_size": 32, "patience": 3},
            {"units": 48, "learning_rate": 0.001, "epochs": 22, "batch_size": 32, "patience": 4},
            {"units": 64, "learning_rate": 0.0005, "epochs": 26, "batch_size": 32, "patience": 5},
        ]

    def _inference_only(
        self, model, scaler_X, scaler_y, feature_cols, feature_values, asset, strategy_name, recent_data
    ):
        latest_sequence = feature_values[-self.SEQ_LEN :]
        latest_scaled = scaler_X.transform(latest_sequence).reshape(1, self.SEQ_LEN, len(feature_cols))
        predicted_close = float(scaler_y.inverse_transform(model.predict(latest_scaled, verbose=0)).ravel()[0])

        last_close = float(recent_data["Close"].iloc[-1])
        last_rsi_value = float(recent_data["RSI"].iloc[-1])
        msg = f"RSI: {last_rsi_value:.2f}, Predicted next Close: {predicted_close:.4f}, Last Close: {last_close:.4f}"

        if last_rsi_value < 30 and predicted_close > last_close:
            decision = "BUY"
        elif last_rsi_value > 70 and predicted_close < last_close:
            decision = "SELL"
        else:
            decision = "HOLD"

        self.log_action(f"{msg} -> {decision} signal", "info")
        trade_summary = self.order_executor.process_signal(asset, decision, last_close, self.risk_manager)
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")
        send_email(
            f"{decision} Signal for {asset} using {strategy_name}",
            f"{msg} -> {decision} signal | trade: {trade_summary}",
            self.config["notification_email"],
        )
        return decision

    def _run_strategy(self, data):
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]

        self.log_action(f"Executing {strategy_name} strategy for {asset} using LSTM model", "info")
        recent_data = data.tail(720).copy()
        if recent_data.empty:
            self.log_action("Recent data empty; skipping execution.", "warning")
            return

        # Normalize timezone, deduplicate, sort index
        if isinstance(recent_data.index, pd.DatetimeIndex) and recent_data.index.tz is None:
            recent_data.index = recent_data.index.tz_localize("UTC")
        if recent_data.index.duplicated().any():
            self.log_action("Duplicate index detected; deduplicating.", "warning")
            recent_data = recent_data[~recent_data.index.duplicated(keep="first")]
        if not recent_data.index.is_monotonic_increasing:
            self.log_action("Index not sorted; sorting now.", "warning")
            recent_data = recent_data.sort_index()

        if self.config.get("use_indicators", True) and "rsi" in self.config.get("indicators", []):
            self.log_action("Calculating RSI indicator...", "info")
            rsi_indicator = RSI(recent_data)
            recent_data["RSI"] = rsi_indicator.calculate()
        else:
            recent_data["RSI"] = recent_data["Close"].pct_change().fillna(0)

        # Additional microstructure features
        if "Volume" not in recent_data.columns:
            self.log_action("Volume column missing; filling Volume with zeros for day-trading features.", "warning")
            recent_data["Volume"] = 0.0

        # Clip returns outliers for stability
        returns = recent_data["Close"].pct_change()
        lower_bound = returns.quantile(0.01)
        upper_bound = returns.quantile(0.99)
        returns_clipped = returns.clip(lower=lower_bound, upper=upper_bound)
        recent_data["Volatility_10"] = returns_clipped.rolling(10).std()

        # ATR for volatility-adjusted signal
        if all(col in recent_data.columns for col in ["High", "Low", "Close"]):
            high_low = recent_data["High"] - recent_data["Low"]
            high_close = (recent_data["High"] - recent_data["Close"].shift()).abs()
            low_close = (recent_data["Low"] - recent_data["Close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            # Clip ATR outliers
            atr_raw = tr.rolling(14).mean()
            atr_upper = atr_raw.quantile(0.99)
            recent_data["ATR_14"] = atr_raw.clip(upper=atr_upper)
        else:
            recent_data["ATR_14"] = 0.0

        feature_cols = ["Close", "RSI", "Volume", "Volatility_10", "ATR_14"]
        recent_data = recent_data.dropna(subset=feature_cols)
        if len(recent_data) <= self.SEQ_LEN:
            self.log_action("Not enough rows to build LSTM sequences.", "warning")
            return

        self._log_feature_stats(recent_data[feature_cols], stage="train/inference window")

        feature_values = recent_data[feature_cols].values
        target_values = recent_data["Close"].values
        X_seq, y_seq = self._create_sequences(feature_values, target_values, self.SEQ_LEN)
        latest_idx = self._latest_index(recent_data)

        persistence = ModelPersistence()
        artifact = persistence.load(strategy_name, is_keras=True)
        if artifact:
            meta = artifact.get("metadata", {})
            saved_at_raw = meta.get("saved_at")
            saved_at = None
            if isinstance(saved_at_raw, str):
                try:
                    saved_at = datetime.fromisoformat(saved_at_raw.replace("Z", "+00:00"))
                except Exception:  # noqa: BLE001
                    saved_at = None
            if saved_at is None and isinstance(saved_at_raw, datetime):
                saved_at = saved_at_raw

            trained_until = meta.get("trained_until")
            model = artifact.get("model")
            scalers = artifact.get("scaler") or {}
            scaler_X = scalers.get("scaler_X")
            scaler_y = scalers.get("scaler_y")

            can_infer = model is not None and scaler_X is not None and scaler_y is not None

            if trained_until == latest_idx and can_infer:
                self.log_action("Loaded persisted LSTM pipeline; skipping retrain.", "info")
                return self._inference_only(
                    model,
                    scaler_X,
                    scaler_y,
                    feature_cols,
                    feature_values,
                    asset,
                    strategy_name,
                    recent_data,
                )

            if saved_at and can_infer:
                saved_at_aware = saved_at if saved_at.tzinfo else saved_at.replace(tzinfo=timezone.utc)
                saved_at_aware = saved_at_aware.astimezone(timezone.utc)
                age_seconds = (datetime.now(tz=timezone.utc) - saved_at_aware).total_seconds()
                if age_seconds < self.MIN_RETRAIN_SECONDS:
                    self.log_action(
                        f"Skipping retrain (cooldown {self.MIN_RETRAIN_SECONDS}s, age={age_seconds:.0f}s); reusing model.",
                        "warning",
                    )
                    return self._inference_only(
                        model,
                        scaler_X,
                        scaler_y,
                        feature_cols,
                        feature_values,
                        asset,
                        strategy_name,
                        recent_data,
                    )

        # Hyperparam tuning with TimeSeriesSplit
        mae_scores = []
        n_splits = min(3, max(2, len(X_seq) // 60))
        tuner_grid = self._build_tuner_grid()
        best_cfg = tuner_grid[0]
        best_cv_mae = None

        if n_splits and len(X_seq) > n_splits:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for cfg in tuner_grid:
                fold_mae = []
                for train_idx, val_idx in tscv.split(X_seq):
                    X_train_raw, X_val_raw = X_seq[train_idx], X_seq[val_idx]
                    y_train_raw, y_val_raw = y_seq[train_idx], y_seq[val_idx]

                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()
                    X_train_scaled = scaler_X.fit_transform(X_train_raw.reshape(-1, len(feature_cols))).reshape(
                        X_train_raw.shape
                    )
                    X_val_scaled = scaler_X.transform(X_val_raw.reshape(-1, len(feature_cols))).reshape(X_val_raw.shape)
                    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
                    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1))

                    model = create_lstm_model(
                        input_shape=(self.SEQ_LEN, len(feature_cols)),
                        units=cfg["units"],
                        learning_rate=cfg["learning_rate"],
                    )
                    model.fit(
                        X_train_scaled,
                        y_train_scaled,
                        validation_data=(X_val_scaled, y_val_scaled),
                        epochs=cfg["epochs"],
                        batch_size=cfg["batch_size"],
                        callbacks=[
                            EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)
                        ],
                        verbose=0,
                    )
                    preds_scaled = model.predict(X_val_scaled, verbose=0)
                    preds = scaler_y.inverse_transform(preds_scaled).ravel()
                    fold_mae.append(mean_absolute_error(y_val_raw, preds))
                median_mae = float(np.median(fold_mae))
                mae_scores.append(median_mae)
                if best_cv_mae is None or median_mae < best_cv_mae:
                    best_cv_mae = median_mae
                    best_cfg = cfg
        else:
            self.log_action("Not enough sequences for walk-forward validation; skipping CV.", "warning")

        # Chronological train/val/test with best cfg
        n = len(X_seq)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        X_train_raw, X_val_raw, X_test_raw = X_seq[:train_end], X_seq[train_end:val_end], X_seq[val_end:]
        y_train_raw, y_val_raw, y_test_raw = y_seq[:train_end], y_seq[train_end:val_end], y_seq[val_end:]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_raw.reshape(-1, len(feature_cols))).reshape(X_train_raw.shape)
        X_val_scaled = scaler_X.transform(X_val_raw.reshape(-1, len(feature_cols))).reshape(X_val_raw.shape)
        X_test_scaled = scaler_X.transform(X_test_raw.reshape(-1, len(feature_cols))).reshape(X_test_raw.shape)
        y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
        y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1))

        model = create_lstm_model(
            input_shape=(self.SEQ_LEN, len(feature_cols)),
            units=best_cfg["units"],
            learning_rate=best_cfg["learning_rate"],
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=best_cfg["patience"], restore_best_weights=True)
        model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=best_cfg["epochs"],
            batch_size=best_cfg["batch_size"],
            callbacks=[early_stopping],
            verbose=0,
        )

        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
        test_mae = mean_absolute_error(y_test_raw, y_pred)
        baseline_naive = X_test_raw[:, -1, 0] if X_test_raw.size else np.array([])
        baseline_mae = mean_absolute_error(y_test_raw, baseline_naive) if baseline_naive.size else None

        walkforward_mae = float(np.median(mae_scores)) if mae_scores else None
        if walkforward_mae is not None:
            self.log_action(f"Walk-forward MAE (median): {walkforward_mae:.4f}", "info")
        self.log_action(f"Test MAE: {test_mae:.4f}", "info")
        if baseline_mae is not None:
            self.log_action(f"Naive baseline MAE (predict last close): {baseline_mae:.4f}", "info")

        # Calculate config signature for drift detection
        config_blob = self.config.model_dump() if hasattr(self.config, "model_dump") else self.config
        config_signature = _build_config_signature(config_blob)

        # Persist model, scalers, metadata
        persistence.save(
            strategy_name,
            model,
            scaler={"scaler_X": scaler_X, "scaler_y": scaler_y},
            metadata={
                "trained_until": latest_idx,
                "feature_columns": feature_cols,
                "seq_len": self.SEQ_LEN,
                "best_cfg": best_cfg,
                "walkforward_mae": walkforward_mae,
                "test_mae": test_mae,
                "train_rows": len(y_train_raw),
                "baseline_mae_last_close": baseline_mae,
                "config_signature": config_signature,
            },
            is_keras=True,
        )

        self._inference_only(model, scaler_X, scaler_y, feature_cols, feature_values, asset, strategy_name, recent_data)

    def _log_feature_stats(self, df, stage: str):
        """Log simple feature stats to monitor stability."""
        try:
            summary = {}
            for col in df.columns:
                series = df[col].dropna()
                summary[col] = {
                    "min": float(series.min()),
                    "median": float(series.median()),
                    "max": float(series.max()),
                    "nan": int(df[col].isna().sum()),
                }
            self.log_action(f"Feature stats ({stage}): {summary}", "info")
        except Exception:  # noqa: BLE001
            return
