from datetime import datetime, timezone

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from trading_bot.indicators.rsi import RSI
from trading_bot.models.lstm_model import create_lstm_model
from trading_bot.strategies.strategy_base import StrategyBase
from trading_bot.utils.email_notifications import send_email
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.strategy_helpers import _build_config_signature, build_persistence_key
from trading_bot.utils.transformers import ReturnOutlierClipper


class DayTradingStrategy(StrategyBase):
    SEQ_LEN = 60
    MIN_ROWS = 60
    MIN_RETRAIN_SECONDS = 3600

    @staticmethod
    def _causal_clip_series(
        series: pd.Series,
        lower_pct: float,
        upper_pct: float,
        min_periods: int = 30,
        min_value: float | None = None,
    ) -> pd.Series:
        lower = series.expanding(min_periods=min_periods).quantile(lower_pct / 100).shift(1)
        upper = series.expanding(min_periods=min_periods).quantile(upper_pct / 100).shift(1)
        if min_value is not None:
            lower = lower.clip(lower=min_value)
        clipped = series.clip(lower, upper)
        mask = lower.isna() | upper.isna()
        return clipped.where(~mask, series)

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

    def _build_decision_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        if self.config.get("use_indicators", True) and "rsi" in self.config.get("indicators", []):
            out["RSI"] = RSI(out).calculate()
        else:
            out["RSI"] = out["Close"].pct_change().fillna(0)
        return out

    def _inference_only(
        self,
        model,
        scaler_X,
        scaler_y,
        feature_cols,
        feature_values,
        asset,
        strategy_name,
        recent_data,
        decision_data: pd.DataFrame | None = None,
        trade_enabled: bool = True,
    ):
        latest_sequence = feature_values[-self.SEQ_LEN :]
        latest_scaled = scaler_X.transform(latest_sequence).reshape(1, self.SEQ_LEN, len(feature_cols))
        predicted_close = float(scaler_y.inverse_transform(model.predict(latest_scaled, verbose=0)).ravel()[0])

        decision_frame = decision_data if decision_data is not None else recent_data
        last_close = float(decision_frame["Close"].iloc[-1])
        last_rsi_value = float(decision_frame["RSI"].iloc[-1])
        msg = f"RSI: {last_rsi_value:.2f}, Predicted next Close: {predicted_close:.4f}, Last Close: {last_close:.4f}"

        if last_rsi_value < 30 and predicted_close > last_close:
            decision = "BUY"
        elif last_rsi_value > 70 and predicted_close < last_close:
            decision = "SELL"
        else:
            decision = "HOLD"

        self.log_action(f"{msg} -> {decision} signal", "info")
        if not trade_enabled:
            self.log_action("Quality gate active; skipping trade execution and notifications.", "warning")
            return decision
        trade_summary = self.order_executor.process_signal(asset, decision, last_close, self.risk_manager)
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")
        send_email(
            f"{decision} Signal for {asset} using {strategy_name}",
            f"{msg} -> {decision} signal | trade: {trade_summary}",
            self.config["notification_email"],
        )
        return decision

    def _quality_gate_blocks(self, test_mae: float | None, baseline_mae: float | None) -> bool:
        if not self.config.get("lstm_quality_gate_enabled", True):
            return False
        if test_mae is None or baseline_mae is None:
            return False
        try:
            test_mae = float(test_mae)
            baseline_mae = float(baseline_mae)
        except (TypeError, ValueError):
            return False
        if baseline_mae <= 0:
            return False
        ratio = self.config.get("lstm_quality_gate_ratio", 1.0) or 1.0
        if ratio <= 0:
            ratio = 1.0
        if test_mae > baseline_mae * ratio:
            self.log_action(
                f"Quality gate failed: test MAE {test_mae:.4f} exceeds "
                f"baseline {baseline_mae:.4f} * {ratio:.2f}.",
                "warning",
            )
            return True
        return False

    def _run_strategy(self, data):
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]
        inference_only = self.config.get("inference_only", False)
        persistence_key = build_persistence_key(
            strategy=strategy_name,
            data_source=self.config.get("data_source"),
            ticker=asset,
            interval=self.config.get("interval"),
        )

        self.log_action(f"Executing {strategy_name} strategy for {asset} using LSTM model", "info")
        recent_data = self._prepare_recent_window(data)
        if recent_data is None:
            return
        decision_data = self._build_decision_frame(recent_data)
        feature_data = ReturnOutlierClipper(
            price_columns=["Close"],
            lower_pct=0.1,
            upper_pct=99.9,
            min_periods=30,
        ).transform(recent_data)
        build = self._build_features(feature_data)
        if build is None:
            return
        feature_cols, feature_values, target_values, latest_idx, X_seq, y_seq, feature_data = build

        persistence = ModelPersistence()
        artifact = persistence.load(persistence_key, is_keras=True)
        meta = artifact.get("metadata", {}) if artifact else {}
        model = artifact.get("model") if artifact else None
        scalers = (artifact.get("scaler") if artifact else {}) or {}
        scaler_X = scalers.get("scaler_X")
        scaler_y = scalers.get("scaler_y")
        can_infer = model is not None and scaler_X is not None and scaler_y is not None
        trained_until = meta.get("trained_until")

        saved_at_raw = meta.get("saved_at")
        saved_at = None
        if isinstance(saved_at_raw, str):
            try:
                saved_at = datetime.fromisoformat(saved_at_raw.replace("Z", "+00:00"))
            except ValueError:
                saved_at = None
        if saved_at is None and isinstance(saved_at_raw, datetime):
            saved_at = saved_at_raw

        trained_until_ts = None
        try:
            trained_until_ts = pd.to_datetime(trained_until)
        except (ValueError, TypeError):
            trained_until_ts = None

        def _reuse_with_reason(reason: str):
            if not can_infer:
                return None
            self.log_action(reason, "info" if "Skipping" in reason or "Loaded" in reason else "warning")
            trade_enabled = not self._quality_gate_blocks(
                meta.get("test_mae"),
                meta.get("baseline_mae_last_close"),
            )
            return self._inference_only(
                model,
                scaler_X,
                scaler_y,
                feature_cols,
                feature_values,
                asset,
                strategy_name,
                feature_data,
                decision_data=decision_data,
                trade_enabled=trade_enabled,
            )

        if inference_only:
            if can_infer:
                return _reuse_with_reason("Inference-only mode: using persisted LSTM pipeline.")
            self.log_action("Inference-only mode but no persisted LSTM available; skipping.", "warning")
            return

        if trained_until == latest_idx and can_infer:
            result = _reuse_with_reason("Loaded persisted LSTM pipeline; no new candles since last training.")
            if result is not None:
                return result

        if trained_until_ts is not None:
            new_rows = recent_data[recent_data.index > trained_until_ts]
            if len(new_rows) < self.SEQ_LEN and can_infer:
                return _reuse_with_reason(
                    f"Only {len(new_rows)} new candles since last training (<{self.SEQ_LEN}); reuse existing model."
                )

        if saved_at and can_infer:
            saved_at_aware = saved_at if saved_at.tzinfo else saved_at.replace(tzinfo=timezone.utc)
            saved_at_aware = saved_at_aware.astimezone(timezone.utc)
            age_seconds = (datetime.now(tz=timezone.utc) - saved_at_aware).total_seconds()
            if age_seconds < self.MIN_RETRAIN_SECONDS:
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except (ImportError, RuntimeError):
                    pass
                return _reuse_with_reason(
                    f"Skipping retrain (cooldown {self.MIN_RETRAIN_SECONDS}s, age={age_seconds:.0f}s); reusing model."
                )

        # Hyperparam tuning with TimeSeriesSplit
        mae_scores = []
        n_splits = min(3, max(2, len(X_seq) // 60))
        tuner_grid = self._build_tuner_grid()
        best_cfg = tuner_grid[0]
        best_cv_mae = None
        try:
            import tensorflow as tf
        except ImportError:  # pragma: no cover - TensorFlow required for LSTM strategy
            tf = None

        if n_splits and len(X_seq) > n_splits:
            tscv = None
            try:
                tscv = TimeSeriesSplit(n_splits=n_splits, gap=self.SEQ_LEN)
            except ValueError as exc:
                self.log_action(f"TimeSeriesSplit with gap={self.SEQ_LEN} not possible: {exc}", "warning")
            if tscv is None:
                self.log_action("Skipping walk-forward CV due to insufficient data after gap.", "warning")
            else:
                for cfg in tuner_grid:
                    fold_mae = []
                    for train_idx, val_idx in tscv.split(X_seq):
                        if tf is not None:
                            tf.keras.backend.clear_session()
                        X_train_raw, X_val_raw = X_seq[train_idx], X_seq[val_idx]
                        y_train_raw, y_val_raw = y_seq[train_idx], y_seq[val_idx]

                        scaler_X = MinMaxScaler()
                        scaler_y = MinMaxScaler()
                        X_train_scaled = scaler_X.fit_transform(X_train_raw.reshape(-1, len(feature_cols))).reshape(
                            X_train_raw.shape
                        )
                        X_val_scaled = scaler_X.transform(X_val_raw.reshape(-1, len(feature_cols))).reshape(
                            X_val_raw.shape
                        )
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
                        preds_scaled = model(X_val_scaled, training=False)
                        preds_scaled = preds_scaled.numpy() if hasattr(preds_scaled, "numpy") else np.asarray(preds_scaled)
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

        y_pred_scaled = model(X_test_scaled, training=False)
        y_pred_scaled = y_pred_scaled.numpy() if hasattr(y_pred_scaled, "numpy") else np.asarray(y_pred_scaled)
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

        trade_enabled = not self._quality_gate_blocks(test_mae, baseline_mae)

        # Calculate config signature for drift detection
        config_blob = self.config.model_dump() if hasattr(self.config, "model_dump") else self.config
        if isinstance(config_blob, dict):
            config_blob = {
                **config_blob,
                "fe_version": "v2",
                "outlier_clip_pct": (0.1, 99.9),
                "outlier_clip_min_periods": 30,
            }
        config_signature = _build_config_signature(config_blob)

        # Persist model, scalers, metadata
        persistence.save(
            persistence_key,
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
                "ticker": self.config.get("ticker"),
                "interval": self.config.get("interval"),
                "period": self.config.get("period"),
                "data_source": self.config.get("data_source"),
            },
            is_keras=True,
        )

        self._inference_only(
            model,
            scaler_X,
            scaler_y,
            feature_cols,
            feature_values,
            asset,
            strategy_name,
            feature_data,
            decision_data=decision_data,
            trade_enabled=trade_enabled,
        )

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
        except (ValueError, TypeError, KeyError):
            return

    def _prepare_recent_window(self, data: pd.DataFrame) -> pd.DataFrame | None:
        window_rows = self.config.get("train_window_rows") or 720
        recent = data.tail(window_rows).copy()
        if recent.empty:
            self.log_action("Recent data empty; skipping execution.", "warning")
            return None
        if isinstance(recent.index, pd.DatetimeIndex) and recent.index.tz is None:
            recent.index = recent.index.tz_localize("UTC")
        if recent.index.duplicated().any():
            self.log_action("Duplicate index detected; deduplicating.", "warning")
            recent = recent[~recent.index.duplicated(keep="first")]
        if not recent.index.is_monotonic_increasing:
            self.log_action("Index not sorted; sorting now.", "warning")
            recent = recent.sort_index()
        return recent

    def _build_features(self, recent_data: pd.DataFrame):
        if self.config.get("use_indicators", True) and "rsi" in self.config.get("indicators", []):
            self.log_action("Calculating RSI indicator...", "info")
            rsi_indicator = RSI(recent_data)
            recent_data["RSI"] = rsi_indicator.calculate()
        else:
            recent_data["RSI"] = recent_data["Close"].pct_change().fillna(0)

        if "Volume" not in recent_data.columns:
            self.log_action("Volume column missing; filling Volume with zeros for day-trading features.", "warning")
            recent_data["Volume"] = 0.0

        returns = recent_data["Close"].pct_change()
        returns_clipped = self._causal_clip_series(
            returns,
            lower_pct=1.0,
            upper_pct=99.0,
            min_periods=30,
            min_value=-0.99,
        )
        recent_data["Volatility_10"] = returns_clipped.rolling(10).std()

        if all(col in recent_data.columns for col in ["High", "Low", "Close"]):
            high_low = recent_data["High"] - recent_data["Low"]
            high_close = (recent_data["High"] - recent_data["Close"].shift()).abs()
            low_close = (recent_data["Low"] - recent_data["Close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_raw = tr.rolling(14).mean()
            atr_upper = atr_raw.expanding(min_periods=30).quantile(0.99).shift(1)
            atr_clipped = atr_raw.clip(upper=atr_upper)
            if atr_upper.isna().any():
                atr_clipped = atr_clipped.where(~atr_upper.isna(), atr_raw)
            recent_data["ATR_14"] = atr_clipped
        else:
            recent_data["ATR_14"] = 0.0

        feature_cols = ["Close", "RSI", "Volume", "Volatility_10", "ATR_14"]
        recent_data = recent_data.dropna(subset=feature_cols)
        if len(recent_data) <= self.SEQ_LEN:
            self.log_action("Not enough rows to build LSTM sequences.", "warning")
            return None

        self._log_feature_stats(recent_data[feature_cols], stage="train/inference window")

        feature_values = recent_data[feature_cols].values
        target_values = recent_data["Close"].values
        X_seq, y_seq = self._create_sequences(feature_values, target_values, self.SEQ_LEN)
        latest_idx = self._latest_index(recent_data)
        return feature_cols, feature_values, target_values, latest_idx, X_seq, y_seq, recent_data
