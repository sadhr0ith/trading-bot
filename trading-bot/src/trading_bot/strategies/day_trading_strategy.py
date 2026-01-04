from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from trading_bot.indicators.rsi import RSI
from trading_bot.models.lstm_model import create_lstm_model
from trading_bot.models.signal import SignalAction
from trading_bot.strategies.strategy_base import StrategyBase
from trading_bot.data_fetcher import fetch_data_online
from trading_bot.utils.email_notifications import send_email
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.proxy_metrics import long_only_proxy
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
        predicted_return = float(scaler_y.inverse_transform(model.predict(latest_scaled, verbose=0)).ravel()[0])

        decision_frame = decision_data if decision_data is not None else recent_data
        last_close = float(decision_frame["Close"].iloc[-1])
        min_edge = self._min_edge()
        msg = (
            f"Predicted return: {predicted_return:.6f}, Last Close: {last_close:.4f}, "
            f"Min edge: {min_edge:.6f}"
        )

        if predicted_return > min_edge:
            decision = "BUY"
        elif predicted_return < -min_edge:
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

    def _min_edge(self) -> float:
        fee = float(self.risk_manager.trading_fee)
        slippage = float(self.config.get("slippage_rate", 0.0002))
        threshold = float(self.config.get("prediction_threshold", 0.0))
        return abs(threshold) + fee + slippage

    def _gate_from_metadata(self, meta: dict) -> bool:
        trade_enabled = not self._quality_gate_blocks(
            meta.get("test_mae"),
            meta.get("baseline_mae"),
        )
        min_hit_rate = self.config.get("ml_min_hit_rate")
        hit_rate = meta.get("hit_rate")
        if trade_enabled and min_hit_rate is not None and hit_rate is not None and hit_rate < min_hit_rate:
            trade_enabled = False
        max_dd_limit = self.config.get("ml_max_drawdown")
        max_dd = meta.get("max_drawdown_proxy_net", meta.get("max_drawdown_proxy"))
        if trade_enabled and max_dd_limit is not None and max_dd is not None and abs(max_dd) > max_dd_limit:
            trade_enabled = False
        pnl_proxy = meta.get("pnl_proxy_net", meta.get("pnl_proxy"))
        if trade_enabled and pnl_proxy is not None and pnl_proxy <= 0:
            trade_enabled = False
        if trade_enabled and not self._benchmark_gate(meta):
            trade_enabled = False
        return trade_enabled

    @staticmethod
    def _hit_rate_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float | None:
        if y_true.size == 0 or y_pred.size == 0:
            return None
        mask = np.abs(y_pred) > threshold
        if not np.any(mask):
            return None
        hits = np.sign(y_true[mask]) == np.sign(y_pred[mask])
        return float(np.mean(hits)) if hits.size else None

    @staticmethod
    def _pnl_proxy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float | None, float | None]:
        if y_true.size == 0 or y_pred.size == 0:
            return None, None
        mask = np.abs(y_pred) > threshold
        if not np.any(mask):
            return None, None
        long_returns = np.where(y_pred[mask] > 0, y_true[mask], 0.0)
        equity = pd.Series((1.0 + long_returns).cumprod())
        pnl_proxy = float(equity.iloc[-1] - 1.0) if not equity.empty else None
        max_dd = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else None
        return pnl_proxy, max_dd

    def _benchmark_gate(self, lstm_meta: dict) -> bool:
        if not self.config.get("lstm_benchmark_enabled", True):
            return True
        benchmark_strategy = self.config.get("lstm_benchmark_strategy", "day_trading_ml")
        if not benchmark_strategy:
            return True

        required = bool(self.config.get("lstm_benchmark_required", True))
        min_delta = float(self.config.get("lstm_benchmark_min_delta", 0.0) or 0.0)

        data_source = self.config.get("data_source")
        ticker = self.config.get("ticker")
        interval = self.config.get("interval")
        benchmark_key = build_persistence_key(
            strategy=benchmark_strategy,
            data_source=data_source,
            ticker=ticker,
            interval=interval,
        )
        benchmark_meta = ModelPersistence().load_metadata(benchmark_key) or {}

        bench_pnl = benchmark_meta.get("pnl_proxy_net", benchmark_meta.get("pnl_proxy"))
        lstm_pnl = lstm_meta.get("pnl_proxy_net", lstm_meta.get("pnl_proxy"))

        if bench_pnl is None or lstm_pnl is None:
            if required:
                self.log_action(
                    f"Quality gate failed: missing benchmark metrics for {benchmark_strategy}.",
                    "warning",
                )
                return False
            return True

        try:
            bench_pnl = float(bench_pnl)
            lstm_pnl = float(lstm_pnl)
        except (TypeError, ValueError):
            if required:
                self.log_action(
                    f"Quality gate failed: invalid benchmark metrics for {benchmark_strategy}.",
                    "warning",
                )
                return False
            return True

        if lstm_pnl <= bench_pnl + min_delta:
            self.log_action(
                f"Quality gate failed: LSTM pnl_proxy_net {lstm_pnl:.4f} not above "
                f"{benchmark_strategy} {bench_pnl:.4f} (+{min_delta:.4f}).",
                "warning",
            )
            return False
        return True

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

        extra_frames = self._load_training_universe()
        if extra_frames:
            extra_sequences = []
            for extra_df in extra_frames:
                extra_recent = self._prepare_recent_window(extra_df)
                if extra_recent is None:
                    continue
                extra_feature_data = ReturnOutlierClipper(
                    price_columns=["Close"],
                    lower_pct=0.1,
                    upper_pct=99.9,
                    min_periods=30,
                ).transform(extra_recent)
                extra_build = self._build_features(extra_feature_data)
                if extra_build is None:
                    continue
                _, _, _, _, X_seq_extra, y_seq_extra, _ = extra_build
                if X_seq_extra.size and y_seq_extra.size:
                    extra_sequences.append((X_seq_extra, y_seq_extra))

            if extra_sequences:
                X_seq = np.concatenate([X_seq] + [seq[0] for seq in extra_sequences])
                y_seq = np.concatenate([y_seq] + [seq[1] for seq in extra_sequences])
                self.log_action(f"Added {len(extra_sequences)} extra assets to LSTM training pool.", "info")

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
            trade_enabled = self._gate_from_metadata(meta)
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

        baseline_zero = float(np.mean(np.abs(y_test_raw))) if y_test_raw.size else None
        if y_test_raw.size > 1:
            baseline_last = mean_absolute_error(y_test_raw[1:], y_test_raw[:-1])
        else:
            baseline_last = None
        baseline_candidates = [val for val in [baseline_zero, baseline_last] if val is not None]
        baseline_mae = min(baseline_candidates) if baseline_candidates else None

        threshold = self._min_edge()
        hit_rate = self._hit_rate_at_threshold(y_test_raw, y_pred, threshold)
        pnl_proxy, max_dd_proxy = self._pnl_proxy(y_test_raw, y_pred, threshold)
        cost_per_side = float(self.risk_manager.trading_fee) + float(self.config.get("slippage_rate", 0.0002))
        proxy_net = long_only_proxy(y_test_raw, y_pred, threshold=threshold, cost_per_side=cost_per_side)
        pnl_proxy_net = proxy_net.get("pnl_proxy")
        max_dd_proxy_net = proxy_net.get("max_drawdown")

        walkforward_mae = float(np.median(mae_scores)) if mae_scores else None
        if walkforward_mae is not None:
            self.log_action(f"Walk-forward MAE (median): {walkforward_mae:.4f}", "info")
        self.log_action(f"Test MAE: {test_mae:.4f}", "info")
        if baseline_zero is not None:
            self.log_action(f"Baseline MAE (predict 0 return): {baseline_zero:.4f}", "info")
        if baseline_last is not None:
            self.log_action(f"Baseline MAE (predict last return): {baseline_last:.4f}", "info")
        if hit_rate is not None:
            self.log_action(f"Hit rate @ threshold: {hit_rate:.2%}", "info")
        if pnl_proxy is not None:
            self.log_action(f"PnL proxy: {pnl_proxy:.4f}", "info")
        if pnl_proxy_net is not None:
            self.log_action(f"PnL proxy (net): {pnl_proxy_net:.4f}", "info")

        trade_enabled = not self._quality_gate_blocks(test_mae, baseline_mae)
        min_hit_rate = self.config.get("ml_min_hit_rate")
        if trade_enabled and min_hit_rate is not None and hit_rate is not None and hit_rate < min_hit_rate:
            trade_enabled = False
            self.log_action(
                f"Quality gate failed: hit rate {hit_rate:.2%} below {min_hit_rate:.2%}.",
                "warning",
            )
        max_dd_limit = self.config.get("ml_max_drawdown")
        dd_value = max_dd_proxy_net if max_dd_proxy_net is not None else max_dd_proxy
        if trade_enabled and max_dd_limit is not None and dd_value is not None:
            if abs(dd_value) > max_dd_limit:
                trade_enabled = False
                self.log_action(
                    f"Quality gate failed: max drawdown {dd_value:.4f} exceeds {max_dd_limit:.4f}.",
                    "warning",
                )
        pnl_value = pnl_proxy_net if pnl_proxy_net is not None else pnl_proxy
        if trade_enabled and pnl_value is not None and pnl_value <= 0:
            trade_enabled = False
            self.log_action("Quality gate failed: PnL proxy <= 0.", "warning")
        if trade_enabled and not self._benchmark_gate({"pnl_proxy_net": pnl_proxy_net, "pnl_proxy": pnl_proxy}):
            trade_enabled = False

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
                "baseline_mae_zero": baseline_zero,
                "baseline_mae_last": baseline_last,
                "baseline_mae": baseline_mae,
                "hit_rate": hit_rate,
                "pnl_proxy": pnl_proxy,
                "max_drawdown_proxy": max_dd_proxy,
                "pnl_proxy_net": pnl_proxy_net,
                "max_drawdown_proxy_net": max_dd_proxy_net,
                "fe_version": "v2",
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

    def _load_training_universe(self) -> list[pd.DataFrame]:
        tickers = self.config.get("training_tickers") or []
        if not tickers:
            return []
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            if ticker == self.config.get("ticker"):
                continue
            df = fetch_data_online(
                source=self.config.get("data_source", "binance"),
                ticker=ticker,
                period=self.config.get("period", "1y"),
                interval=self.config.get("interval", "1h"),
                cache_ttl_seconds=self.config.get("cache_ttl_seconds"),
            )
            if df is None or df.empty:
                self.log_action(f"No training data for {ticker}; skipping.", "warning")
                continue
            frames.append(df)
        return frames

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

        horizon = int(self.config.get("return_horizon", 1))
        recent_data["target"] = recent_data["Close"].pct_change(horizon).shift(-horizon)
        recent_data = recent_data.dropna(subset=feature_cols + ["target"])
        if len(recent_data) <= self.SEQ_LEN:
            self.log_action("Not enough rows to build LSTM sequences after target alignment.", "warning")
            return None
        feature_values = recent_data[feature_cols].values
        target_values = recent_data["target"].values
        X_seq, y_seq = self._create_sequences(feature_values, target_values, self.SEQ_LEN)
        latest_idx = self._latest_index(recent_data)
        return feature_cols, feature_values, target_values, latest_idx, X_seq, y_seq, recent_data

    def _compute_signal_action(
        self,
        data: pd.DataFrame,
    ) -> tuple[SignalAction, float | None, float | None, dict[str, Any]] | None:
        """Compute signal action for multi-strategy mode."""
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]
        persistence_key = build_persistence_key(
            strategy=strategy_name,
            data_source=self.config.get("data_source"),
            ticker=asset,
            interval=self.config.get("interval"),
        )

        persistence = ModelPersistence()
        artifact = persistence.load(persistence_key, is_keras=True)
        if not artifact:
            return (SignalAction.HOLD, None, None, {"reason": "no_model"})

        model = artifact.get("model")
        scalers = artifact.get("scaler") or {}
        scaler_X = scalers.get("scaler_X")
        scaler_y = scalers.get("scaler_y")
        meta = artifact.get("metadata", {})

        if model is None or scaler_X is None or scaler_y is None:
            return (SignalAction.HOLD, None, None, {"reason": "incomplete_artifact"})

        # Check quality gate
        trade_enabled = self._gate_from_metadata(meta)
        if not trade_enabled:
            return (SignalAction.HOLD, None, None, {"reason": "quality_gate_failed"})

        # Prepare features
        recent_data = self._prepare_recent_window(data)
        if recent_data is None:
            return (SignalAction.HOLD, None, None, {"reason": "insufficient_data"})

        feature_data = ReturnOutlierClipper(
            price_columns=["Close"],
            lower_pct=0.1,
            upper_pct=99.9,
            min_periods=30,
        ).transform(recent_data)
        build = self._build_features(feature_data)
        if build is None:
            return (SignalAction.HOLD, None, None, {"reason": "feature_build_failed"})

        feature_cols, feature_values, _, _, _, _, _ = build

        # Run inference
        try:
            latest_sequence = feature_values[-self.SEQ_LEN:]
            latest_scaled = scaler_X.transform(latest_sequence).reshape(1, self.SEQ_LEN, len(feature_cols))
            predicted_return = float(scaler_y.inverse_transform(model.predict(latest_scaled, verbose=0)).ravel()[0])
        except (ValueError, TypeError) as exc:
            return (SignalAction.HOLD, None, None, {"reason": f"inference_failed: {exc}"})

        min_edge = self._min_edge()
        if predicted_return > min_edge:
            action = SignalAction.BUY
            reason = "positive_predicted_return"
        elif predicted_return < -min_edge:
            action = SignalAction.SELL
            reason = "negative_predicted_return"
        else:
            action = SignalAction.HOLD
            reason = "within_threshold"

        metadata = {
            "strategy_type": "day_trading_lstm",
            "predicted_return": predicted_return,
            "min_edge": min_edge,
            "reason": reason,
        }

        return (action, None, predicted_return, metadata)
