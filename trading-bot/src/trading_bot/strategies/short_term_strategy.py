from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

from trading_bot.core.exceptions import InsufficientDataError
from trading_bot.models.signal import SignalAction
from trading_bot.utils.email_notifications import send_email
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.strategy_helpers import build_persistence_key, train_or_load_pipeline
from trading_bot.utils.transformers import (
    CalendarFeatureTransformer,
    FeatureSelector,
    IndicatorLagTransformer,
    IndicatorTransformer,
    LagFeatureTransformer,
    ReturnOutlierClipper,
    ReturnFeatureTransformer,
    RollingStatsTransformer,
)

from .strategy_base import StrategyBase

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class ShortTermStrategy(StrategyBase):
    """
    Short-term strategy using XGBoost for 5-day return prediction.

    Features:
    - Lag features: Close prices at lags [1, 3, 5, 10]
    - Return features: Price returns over [1, 3, 5, 10] periods
    - Rolling statistics: SMA, Volatility over [5, 10] windows
    - Calendar features: day_of_week, month, is_month_start, is_month_end
    - Optional indicators: MACD, RSI, ADX with lag features

    Data Requirements:
    - Minimum 25 rows required for inference (lag/rolling transformers need historical context)
    - Minimum 50 rows recommended for training
    - Uses TimeSeriesSplit for walk-forward cross-validation

    Inference:
    - Predicts 5-day ahead returns using last 25 rows (20 for context + 5 buffer)
    - Decision threshold: ±0.5% with MACD confirmation
    """

    MIN_ROWS = 25
    TARGET_HORIZON = 5

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

        self.log_action(f"Executing short-term strategy for {asset} using XGBoost", "info")
        horizon = self.TARGET_HORIZON
        data["target"] = data["Close"].pct_change(horizon).shift(-horizon)

        indicator_names = self._indicator_set()
        feature_columns, indicator_cols, base_feature_columns = self._build_feature_columns(data, indicator_names)
        has_volume = "Volume" in data.columns

        # Validate and split data
        MIN_INFERENCE_ROWS = self.config.get("min_inference_rows", 25)
        if len(data) < MIN_INFERENCE_ROWS:
            raise InsufficientDataError(
                f"Need at least {MIN_INFERENCE_ROWS} rows for inference, got {len(data)}. "
                f"Lag/rolling features require historical context."
            )

        if inference_only:
            data_inference = data.iloc[-MIN_INFERENCE_ROWS:].drop(columns=["target"])
            return self._run_inference_only(
                data, data_inference, feature_columns, strategy_name, asset, persistence_key
            )

        data_train, data_inference = self._split_train_inference(data, MIN_INFERENCE_ROWS, horizon)

        if len(data_train) < 50:
            raise InsufficientDataError(
                f"Not enough training data after removing missing targets: got {len(data_train)}, need at least 50"
            )

        if len(data_train) < 60:
            self.log_action("Small dataset detected (<60 rows); using ElasticNet fallback for sanity.", "warning")
            return self._run_fallback_elasticnet(
                data_train, data_inference, feature_columns, base_feature_columns, indicator_names, persistence_key
            )

        # Train model and get predictions
        pipeline, baseline_mae, elasticnet_baseline_mae, cv_mae = self._train_xgb_model(
            data_train,
            strategy_name,
            persistence_key,
            feature_columns,
            indicator_names,
            indicator_cols,
            base_feature_columns,
            has_volume,
        )

        if pipeline is None:
            return

        # Execute prediction and trade
        self._predict_and_execute_trade(
            pipeline,
            data,
            data_inference,
            feature_columns,
            asset,
            strategy_name,
            baseline_mae,
            elasticnet_baseline_mae,
            cv_mae,
        )

    def _train_xgb_model(
        self,
        data_train: pd.DataFrame,
        strategy_name: str,
        persistence_key: str,
        feature_columns: list[str],
        indicator_names: set[str],
        indicator_cols: list[str],
        base_feature_columns: list[str],
        has_volume: bool,
    ) -> tuple[Any, float, float | None, float | None]:
        """Train XGBoost model with hyperparameter tuning.

        Returns:
            Tuple of (pipeline, baseline_mae, elasticnet_baseline_mae, cv_mae)
        """
        X_train = data_train.drop(columns=["target"])
        y_train = data_train["target"]
        baseline_mae = float(np.mean(np.abs(y_train)))

        persistence = ModelPersistence()
        search_space = self._get_xgb_search_space()

        # Create pipeline factory with captured parameters
        def pipeline_factory():
            return self._build_full_pipeline(indicator_names, indicator_cols, base_feature_columns, has_volume)

        # Create tuner function
        tuner_fn = self._create_tuner(search_space) if search_space else None

        # Evaluate baseline
        elasticnet_baseline_mae = self._evaluate_elasticnet_baseline(
            X_train, y_train, indicator_names, indicator_cols, base_feature_columns, has_volume
        )

        # Train or load pipeline
        pipeline, cv_mae = train_or_load_pipeline(
            key=persistence_key,
            persistence=persistence,
            pipeline_factory=pipeline_factory,
            X=X_train,
            y=y_train,
            feature_columns=feature_columns,
            split_divisor=50,
            metadata={
                "baseline_mae": baseline_mae,
                "train_rows": len(y_train),
                "baseline_elasticnet_mae": elasticnet_baseline_mae,
                "search_space": list(search_space.keys()),
                "ticker": self.config.get("ticker"),
                "interval": self.config.get("interval"),
                "period": self.config.get("period"),
                "data_source": self.config.get("data_source"),
            },
            config_signature_data={
                "model": "short_term_xgb_v1",
                "search_space": search_space,
                "fe_version": "v4",
                "missing_policy": "ffill_zero",
                "missing_flags": True,
                "indicators_in_pipeline": True,
                "outlier_clip_pct": (0.1, 99.9),
                "outlier_clip_min_periods": 30,
                "cv_gap": 5,
                "ticker": self.config.get("ticker"),
                "interval": self.config.get("interval"),
                "period": self.config.get("period"),
                "data_source": self.config.get("data_source"),
            },
            gap=5,
            tuner=tuner_fn,
        )

        return pipeline, baseline_mae, elasticnet_baseline_mae, cv_mae

    def _predict_and_execute_trade(
        self,
        pipeline: Any,
        data: pd.DataFrame,
        data_inference: pd.DataFrame,
        feature_columns: list[str],
        asset: str,
        strategy_name: str,
        baseline_mae: float,
        elasticnet_baseline_mae: float | None,
        cv_mae: float | None,
    ) -> None:
        """Run prediction and execute trade decision."""
        # Log metrics
        if cv_mae is not None:
            self.log_action(f"Cross-validated MAE (walk-forward): {cv_mae:.4f}", "info")
        self.log_action(f"Baseline MAE (predict 0 return): {baseline_mae:.4f}", "info")
        if elasticnet_baseline_mae is not None:
            self.log_action(f"ElasticNet baseline MAE: {elasticnet_baseline_mae:.4f}", "info")

        self._log_feature_importance(pipeline, feature_columns, top_n=10)

        # Validate pipeline
        try:
            check_is_fitted(pipeline)
        except NotFittedError as exc:
            self.log_action(f"Pipeline not fitted or invalid: {exc}", "error")
            return

        # Debug NaN detection
        self._debug_nan_features(pipeline, data_inference)

        # Predict
        predictions = pipeline.predict(data_inference)
        predicted_return = float(predictions[-1])
        self._log_feature_stats_for_inference(pipeline, data_inference)

        # Make decision and execute trade
        buy_threshold, sell_threshold = self._get_adaptive_thresholds(data)
        decision_data = self._build_decision_frame(data)
        decision, reason, msg, _, _ = self._decide_signal(predicted_return, decision_data, buy_threshold, sell_threshold)

        self.log_action(f"{msg} -> {decision} ({reason})", "info" if decision != "HOLD" else "warning")
        trade_summary = self.order_executor.process_signal(asset, decision, data["Close"].iloc[-1], self.risk_manager)
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")

        send_email(
            f"{decision} Signal for {asset} using {strategy_name}",
            f"{msg} -> {decision} ({reason}) | trade: {trade_summary}",
            self.config["notification_email"],
        )

    def _build_fe_pipeline(
        self,
        indicator_names: set[str],
        indicator_cols: list[str],
        base_feature_columns: list[str],
        has_volume: bool,
    ) -> Pipeline:
        """Build feature engineering pipeline."""
        fe_steps = [
            ("outlier_clip", self._build_outlier_clipper()),
            ("indicators", IndicatorTransformer(
                indicators=indicator_names,
                use_indicators=self.config.get("use_indicators", True),
            )),
            ("lag_features", LagFeatureTransformer(columns=["Close"], lags=[1, 3, 5, 10])),
            ("return_features", ReturnFeatureTransformer(price_col="Close", periods=[1, 3, 5, 10])),
            ("rolling_stats", RollingStatsTransformer(
                price_col="Close",
                volume_col="Volume" if has_volume else None,
                windows=[5, 10],
            )),
            ("calendar_features", CalendarFeatureTransformer()),
        ]

        if indicator_cols:
            fe_steps.append((
                "indicator_lags",
                IndicatorLagTransformer(
                    indicator_columns=indicator_cols,
                    lags=[1],
                    column_mapping=self._indicator_column_mapping(),
                ),
            ))

        fe_steps.append((
            "feature_selector",
            FeatureSelector(feature_columns=base_feature_columns, handle_missing="ffill", add_missing_flags=True),
        ))

        return Pipeline(fe_steps)

    def _build_full_pipeline(
        self,
        indicator_names: set[str],
        indicator_cols: list[str],
        base_feature_columns: list[str],
        has_volume: bool,
    ) -> Pipeline:
        """Build complete ML pipeline: FE -> Scaler -> XGBoost."""
        return Pipeline([
            ("feature_engineering", self._build_fe_pipeline(indicator_names, indicator_cols, base_feature_columns, has_volume)),
            ("scaler", StandardScaler()),
            ("model", self._create_xgb_model()),
        ])

    def _create_xgb_model(self) -> BaseEstimator:
        """Create XGBoost model with fallback to DummyRegressor."""
        try:
            return XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                max_depth=6,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
            )
        except (ValueError, TypeError) as exc:
            self.log_action(f"Falling back to DummyRegressor due to model init error: {exc}", "warning")
            return DummyRegressor(strategy="mean")

    def _get_xgb_search_space(self) -> dict[str, list]:
        """Get hyperparameter search space for XGBoost."""
        model = self._create_xgb_model()
        if isinstance(model, DummyRegressor):
            return {}
        return {
            "model__max_depth": [4, 5, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__n_estimators": [200, 300, 500],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__min_child_weight": [1, 5, 10],
        }

    def _create_tuner(self, search_space: dict[str, list]):
        """Create tuner function for hyperparameter search."""
        def tuner(pipeline, X, y, cv, scoring):
            n_iter = 4 if len(X) < 150 else 8
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=search_space,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=self.seed,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y)
            cv_metric = float(abs(search.best_score_)) if "neg_mean_absolute_error" in scoring else float(search.best_score_)
            tuner_meta = {"tuner": "RandomizedSearchCV", "best_params": search.best_params_, "n_iter": search.n_iter}
            return search.best_estimator_, cv_metric, tuner_meta
        return tuner

    def _evaluate_elasticnet_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        indicator_names: set[str],
        indicator_cols: list[str],
        base_feature_columns: list[str],
        has_volume: bool,
    ) -> float | None:
        """Evaluate ElasticNet baseline for comparison."""
        if len(X_train) < 100:
            return None
        try:
            baseline_pipeline = Pipeline([
                ("feature_engineering", self._build_fe_pipeline(indicator_names, indicator_cols, base_feature_columns, has_volume)),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=self.seed)),
            ])
            baseline_tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train) // 50)), gap=5)
            baseline_scores = cross_val_score(
                baseline_pipeline, X_train, y_train, cv=baseline_tscv, scoring="neg_mean_absolute_error", n_jobs=-1
            )
            return float(np.mean(np.abs(baseline_scores)))
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            self.log_action(f"ElasticNet baseline evaluation failed: {exc}", "warning")
            return None

    def _debug_nan_features(self, pipeline: Any, data_inference: pd.DataFrame) -> None:
        """Debug log NaN features in inference data."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        try:
            X_inference = pipeline.named_steps["feature_engineering"].transform(data_inference)
            if np.isnan(X_inference[-1]).any():
                nan_indices = np.where(np.isnan(X_inference[-1]))[0]
                self.logger.warning(f"NaN features detected in inference row: indices {nan_indices}")
        except (ValueError, TypeError, KeyError) as e:
            self.logger.debug(f"Could not check for NaN features: {e}")

    def _build_feature_columns(self, data, indicator_names):
        """Define the feature set based on config and available columns."""
        base_features: list[str] = [
            "Close_lag_1",
            "Close_lag_3",
            "Close_lag_5",
            "Close_lag_10",
            "Return_lag_1",
            "Return_lag_3",
            "Return_lag_5",
            "Return_lag_10",
            "SMA_5",
            "SMA_10",
            "Volatility_5",
            "Volatility_10",
            "day_of_week",
            "month",
            "is_month_start",
            "is_month_end",
        ]

        has_volume = "Volume" in data.columns
        if has_volume:
            base_features.extend(["Volume_MA_5", "Volume_MA_10"])

        indicator_cols: list[str] = []
        if "macd" in indicator_names:
            indicator_cols.extend(["MACD", "Signal"])
            base_features.extend(["MACD_lag_1", "Signal_lag_1"])
        if "rsi" in indicator_names:
            indicator_cols.append("RSI")
            base_features.append("RSI_lag_1")
        if "adx" in indicator_names:
            indicator_cols.extend(["ADX", "Plus_DI", "Minus_DI"])
            base_features.extend(["ADX_lag_1", "Plus_DI_lag_1", "Minus_DI_lag_1"])

        missing_flags = [f"{col}_missing" for col in base_features]
        feature_columns = base_features + missing_flags

        return feature_columns, indicator_cols, base_features

    def _indicator_set(self) -> set[str]:
        indicators = self.config.get("indicators", [])
        if not self.config.get("use_indicators", True):
            return set()
        if isinstance(indicators, str):
            indicators = [indicators]
        return {str(ind).lower() for ind in indicators}

    def _build_outlier_clipper(self) -> ReturnOutlierClipper:
        return ReturnOutlierClipper(
            price_columns=["Close"],
            lower_pct=0.1,
            upper_pct=99.9,
            min_periods=30,
        )

    def _build_decision_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        indicator_names = self._indicator_set()
        if not indicator_names:
            return data
        clipped = self._build_outlier_clipper().transform(data)
        return IndicatorTransformer(
            indicators=indicator_names,
            use_indicators=True,
        ).transform(clipped)

    def _indicator_column_mapping(self):
        return {
            "DI+": "Plus_DI",
            "DI-": "Minus_DI",
            "+DI": "Plus_DI",
            "-DI": "Minus_DI",
        }

    def _log_feature_stats_for_inference(self, pipeline, raw_df):
        """Log summary stats for the transformed inference row to spot drift/NaN."""
        try:
            fe = pipeline.named_steps.get("feature_engineering")
            if fe is None:
                return
            X_inf = fe.transform(raw_df)
            last_row = X_inf[-1]
            nan_count = int(np.isnan(last_row).sum())
            summary = {
                "nan_features": nan_count,
                "min": float(np.nanmin(last_row)),
                "median": float(np.nanmedian(last_row)),
                "max": float(np.nanmax(last_row)),
            }
            self.log_action(f"Inference feature stats: {summary}", "info")
        except (ValueError, TypeError, KeyError):
            return

    def _run_fallback_elasticnet(
        self,
        data_train,
        data_inference,
        feature_columns,
        base_feature_columns,
        indicator_names,
        persistence_key: str,
    ):
        """Fallback path for tiny datasets: use ElasticNet only."""
        X_train = data_train.drop(columns=["target"])
        y_train = data_train["target"]
        fe_steps = [
            (
                "outlier_clip",
                self._build_outlier_clipper(),
            ),
            (
                "indicators",
                IndicatorTransformer(
                    indicators=indicator_names,
                    use_indicators=self.config.get("use_indicators", True),
                ),
            ),
            ("lag_features", LagFeatureTransformer(columns=["Close"], lags=[1, 3, 5, 10])),
            ("return_features", ReturnFeatureTransformer(price_col="Close", periods=[1, 3, 5, 10])),
            (
                "rolling_stats",
                RollingStatsTransformer(
                    price_col="Close",
                    volume_col="Volume" if "Volume" in X_train.columns else None,
                    windows=[5, 10],
                ),
            ),
            ("calendar_features", CalendarFeatureTransformer()),
            (
                "feature_selector",
                FeatureSelector(
                    feature_columns=base_feature_columns,
                    handle_missing="ffill",
                    add_missing_flags=True,
                ),
            ),
        ]
        pipeline = Pipeline(
            [
                ("feature_engineering", Pipeline(fe_steps)),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=self.seed)),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(data_inference)
        predicted_return = float(preds[-1])
        self.log_action(f"ElasticNet fallback predicted 5-day return: {predicted_return:.4f}", "info")
        hold_threshold = 0.005
        if predicted_return > hold_threshold:
            decision = "BUY"
            reason = "positive expected return (fallback)"
        elif predicted_return < -hold_threshold:
            decision = "SELL"
            reason = "negative expected return (fallback)"
        else:
            decision = "HOLD"
            reason = "signal below threshold (fallback)"
        trade_summary = self.order_executor.process_signal(
            self.config["ticker"], decision, data_train["Close"].iloc[-1], self.risk_manager
        )
        self.log_action(f"Fallback trade summary: {trade_summary}", "info")
        send_email(
            f"{decision} Signal (fallback ElasticNet) for {self.config['ticker']} using {self.config['strategy']}",
            f"{predicted_return:.4f} -> {decision} ({reason}) | trade: {trade_summary}",
            self.config["notification_email"],
        )
        try:
            from trading_bot.utils.strategy_helpers import _build_config_signature

            config_signature = _build_config_signature(
                {
                    "model": "elasticnet_fallback",
                    "fe_version": "v2",
                    "missing_policy": "ffill_zero",
                    "missing_flags": True,
                    "indicators_in_pipeline": True,
                    "outlier_clip_pct": (0.1, 99.9),
                    "outlier_clip_min_periods": 30,
                }
            )
            persistence = ModelPersistence()
            persistence.save(
                persistence_key,
                pipeline,
                scaler=None,
                metadata={
                    "trained_until": str(data_train.index.max()) if hasattr(data_train, "index") else None,
                    "feature_columns": feature_columns,
                    "model": "short_term_elasticnet_fallback",
                    "fe_version": "v2",
                    "config_signature": config_signature,
                    "ticker": self.config.get("ticker"),
                    "interval": self.config.get("interval"),
                    "period": self.config.get("period"),
                    "data_source": self.config.get("data_source"),
                },
            )
        except (OSError, pickle.PicklingError, ValueError) as exc:
            self.log_action(f"Failed to persist fallback model: {exc}", "warning")
        return decision

    def _split_train_inference(self, data, min_inference_rows, horizon: int):
        split_idx = len(data) - min_inference_rows
        train_end = split_idx - horizon
        if train_end <= 0:
            raise InsufficientDataError(
                f"Not enough rows to separate training from inference with horizon {horizon}."
            )
        data_train = data.iloc[:train_end].copy()
        data_train = data_train[data_train["target"].notna()]
        data_inference = data.iloc[-min_inference_rows:].drop(columns=["target"])
        assert len(data_train) + len(data_inference) <= len(data), "Train/inference overlap detected!"
        self.log_action(
            f"Training on {len(data_train)} rows, reserving last {min_inference_rows} for inference "
            f"with a {horizon}-row leakage gap",
            "info",
        )
        return data_train, data_inference

    def _decide_signal(self, predicted_return, data, buy_threshold, sell_threshold):
        last_macd = data["MACD"].iloc[-1] if "MACD" in data.columns else None
        last_signal = data["Signal"].iloc[-1] if "Signal" in data.columns else None
        msg = f"Predicted 5-day return: {predicted_return:.4f}"
        if last_macd is not None and last_signal is not None:
            msg += f", MACD: {last_macd:.4f}, Signal: {last_signal:.4f}"
        if predicted_return > buy_threshold and (last_macd is None or last_macd > last_signal):
            decision = "BUY"
            reason = (
                "positive expected return with MACD confirmation"
                if last_macd is not None
                else "positive expected return"
            )
        elif predicted_return < sell_threshold and (last_macd is None or last_macd < last_signal):
            decision = "SELL"
            reason = (
                "negative expected return with MACD confirmation"
                if last_macd is not None
                else "negative expected return"
            )
        else:
            decision = "HOLD"
            reason = "signal below threshold or indicators not aligned"
        return decision, reason, msg, last_macd, last_signal

    def _run_inference_only(self, data, data_inference, feature_columns, strategy_name, asset, persistence_key: str):
        persistence = ModelPersistence()
        try:
            artifact = persistence.load(persistence_key)
        except (OSError, pickle.UnpicklingError, ValueError) as exc:
            self.log_action(f"Inference-only mode but failed to load persisted pipeline: {exc}", "warning")
            return
        if not artifact or not artifact.get("model"):
            self.log_action("Inference-only mode but no persisted pipeline available.", "warning")
            return
        pipeline = artifact.get("model")
        meta = artifact.get("metadata") or {}
        try:
            check_is_fitted(pipeline)
        except NotFittedError as exc:
            self.log_action(f"Persisted pipeline not fitted or invalid: {exc}", "error")
            return
        try:
            predictions = pipeline.predict(data_inference)
        except (ValueError, TypeError, NotFittedError) as exc:
            missing_cols = meta.get("feature_columns") or feature_columns
            self.log_action(
                f"Inference-only: failed to run pipeline predict (expected base/feature columns like {missing_cols}): {exc}",
                "error",
            )
            return
        predicted_return = float(predictions[-1])
        # Use adaptive thresholds if enabled
        buy_threshold, sell_threshold = self._get_adaptive_thresholds(data)
        decision_data = self._build_decision_frame(data)
        decision, reason, msg, last_macd, last_signal = self._decide_signal(
            predicted_return, decision_data, buy_threshold, sell_threshold
        )
        if last_macd is not None and last_signal is not None:
            msg = msg.replace("return:", "return (inference-only):")
        self.log_action(f"{msg} -> {decision} ({reason})", "info" if decision != "HOLD" else "warning")
        trade_summary = self.order_executor.process_signal(asset, decision, data["Close"].iloc[-1], self.risk_manager)
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")
        send_email(
            f"{decision} Signal for {asset} using {strategy_name}",
            f"{msg} -> {decision} ({reason}) | trade: {trade_summary}",
            self.config["notification_email"],
        )

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

        # Try to load and run inference
        persistence = ModelPersistence()
        try:
            artifact = persistence.load(persistence_key)
            if not artifact or not artifact.get("model"):
                return (SignalAction.HOLD, None, None, {"reason": "no_model"})

            pipeline = artifact.get("model")
            check_is_fitted(pipeline)

            MIN_INFERENCE_ROWS = self.config.get("min_inference_rows", 25)
            if len(data) < MIN_INFERENCE_ROWS:
                return (SignalAction.HOLD, None, None, {"reason": "insufficient_data"})

            data_inference = data.iloc[-MIN_INFERENCE_ROWS:]
            predictions = pipeline.predict(data_inference)
            predicted_return = float(predictions[-1])
        except (OSError, pickle.UnpicklingError, ValueError, NotFittedError, KeyError) as exc:
            return (SignalAction.HOLD, None, None, {"reason": f"inference_failed: {exc}"})

        buy_threshold, sell_threshold = self._get_adaptive_thresholds(data)
        decision_data = self._build_decision_frame(data)
        decision, reason, _, _, _ = self._decide_signal(predicted_return, decision_data, buy_threshold, sell_threshold)

        action_map = {"BUY": SignalAction.BUY, "SELL": SignalAction.SELL, "HOLD": SignalAction.HOLD}
        action = action_map.get(decision, SignalAction.HOLD)

        metadata = {
            "strategy_type": "short_term",
            "predicted_return": predicted_return,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "reason": reason,
        }

        return (action, None, predicted_return, metadata)
