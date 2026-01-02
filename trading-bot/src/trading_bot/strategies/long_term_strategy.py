from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from trading_bot.utils.email_notifications import send_email
from trading_bot.utils.feature_engineering import build_lag_feature_columns, create_forward_return_target
from trading_bot.utils.transformers import (
    FeatureSelector,
    IndicatorTransformer,
    LagReturnDrawdownTransformer,
    ReturnOutlierClipper,
)
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.strategy_helpers import build_persistence_key, train_or_load_pipeline

from .strategy_base import StrategyBase


class LongTermStrategy(StrategyBase):
    """Long-term (50-day) RandomForest using long lags, return/vol/drawdown features, indicators, and correlation pruning."""

    MIN_ROWS: int = 50

    def _run_strategy(self, data: pd.DataFrame) -> None:
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]
        inference_only = self.config.get("inference_only", False)
        persistence_key = build_persistence_key(
            strategy=strategy_name,
            data_source=self.config.get("data_source"),
            ticker=asset,
            interval=self.config.get("interval"),
        )

        self.log_action(f"Executing long-term strategy for {asset} using Random Forest and SMA/EMA", "info")
        raw_data = data
        indicator_names = self._indicator_set()

        lags = [20, 60, 120, 250]
        clipper = self._build_outlier_clipper()
        feature_columns = build_lag_feature_columns(lags)
        feature_columns.extend(self._indicator_feature_columns(indicator_names))

        fe_pipeline = Pipeline(
            steps=[
                ("outlier_clip", clipper),
                (
                    "indicators",
                    IndicatorTransformer(
                        indicators=indicator_names,
                        use_indicators=self.config.get("use_indicators", True),
                    ),
                ),
                ("lag_features", LagReturnDrawdownTransformer(price_col="Close", lags=lags)),
                ("feature_selector", FeatureSelector(feature_columns=feature_columns, handle_missing="drop")),
            ]
        )

        features = fe_pipeline.transform(data)
        clean_rows = features.dropna()
        if clean_rows.empty:
            self.log_action("Not enough complete rows to generate inference features.", "warning")
            return
        inference_row = clean_rows.tail(1)
        try:
            inference_row = inference_row[feature_columns]
        except KeyError as exc:
            self.log_action(f"Inference row missing expected feature columns: {exc}", "warning")
            return

        if inference_only:
            features = clean_rows[feature_columns]
            y_aligned = pd.Series(np.zeros(len(features)), index=features.index)
        else:
            target_frame = create_forward_return_target(clipper.transform(data), horizon=50)
            target = target_frame["target"]
            combined = features.join(target.rename("target")).dropna()
            if combined.empty or len(combined) < 50:
                self.log_action("Not enough data after feature engineering; skipping execution.", "warning")
                return
            features = combined[feature_columns]
            y_aligned = combined["target"]

        if not inference_only:
            # Remove highly correlated / duplicate features to reduce redundancy
            features, dropped_cols = self._deduplicate_features(features)
            if dropped_cols:
                self.log_action(f"Removed highly correlated features (r>0.999): {sorted(dropped_cols)}", "info")
            feature_columns = list(features.columns)
            try:
                inference_row = inference_row[feature_columns]
            except KeyError as exc:
                self.log_action(f"Inference row missing expected feature columns: {exc}", "warning")
                return
        baseline_mae = float(np.mean(np.abs(y_aligned)))
        elasticnet_baseline_mae = None
        if not inference_only:
            try:
                enet_pipeline = Pipeline(
                    steps=[
                        ("preprocess", ColumnTransformer([("num", StandardScaler(), feature_columns)], remainder="drop")),
                        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=self.seed)),
                    ],
                )
                baseline_tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(features) // 60)), gap=50)
                baseline_scores = cross_val_score(
                    enet_pipeline,
                    features,
                    y_aligned,
                    cv=baseline_tscv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                elasticnet_baseline_mae = float(np.mean(np.abs(baseline_scores)))
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                self.log_action(f"ElasticNet baseline evaluation failed: {exc}", "warning")

        # Pipeline factory
        persistence = ModelPersistence()
        pipeline_factory = lambda: Pipeline(
            steps=[
                ("preprocess", ColumnTransformer([("num", StandardScaler(), feature_columns)], remainder="drop")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=12,
                        random_state=self.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        if inference_only:
            try:
                pipeline, _ = train_or_load_pipeline(
                    key=persistence_key,
                    persistence=persistence,
                    pipeline_factory=pipeline_factory,  # unused in load_only
                    X=features,
                    y=y_aligned,
                    feature_columns=feature_columns,
                    load_only=True,
                    config_signature_data={
                        "model": "long_term_rf_v1",
                        "fe_version": "v3",
                        "indicators_in_pipeline": True,
                        "outlier_clip_pct": (0.1, 99.9),
                        "outlier_clip_min_periods": 30,
                        "cv_gap": 50,
                        "ticker": self.config.get("ticker"),
                        "interval": self.config.get("interval"),
                        "period": self.config.get("period"),
                        "data_source": self.config.get("data_source"),
                    },
                )
            except (OSError, pickle.UnpicklingError, ValueError) as exc:
                self.log_action(f"Inference-only mode but no persisted pipeline available: {exc}", "warning")
                return
            meta = persistence.load(persistence_key) or {}
            persisted_cols = (meta.get("metadata") or {}).get("feature_columns") or feature_columns
            try:
                check_is_fitted(pipeline)
            except NotFittedError as exc:
                self.log_action(f"Persisted pipeline not fitted or invalid: {exc}", "error")
                return
            try:
                inference_slice = inference_row[persisted_cols]
            except KeyError as exc:
                self.log_action(f"Inference-only: missing persisted feature columns: {exc}", "error")
                return
            predicted_return = float(pipeline.predict(inference_slice)[0])
            self._log_feature_importance(pipeline, persisted_cols)
        else:
            # Train or load pipeline (unified persistence)
            pipeline, cv_mae = train_or_load_pipeline(
                key=persistence_key,
                persistence=persistence,
                pipeline_factory=pipeline_factory,
                X=features,
                y=y_aligned,
                feature_columns=feature_columns,
                split_divisor=60,
                metadata={
                    "baseline_mae": baseline_mae,
                    "baseline_elasticnet_mae": elasticnet_baseline_mae,
                    "train_rows": len(y_aligned),
                    "ticker": self.config.get("ticker"),
                    "interval": self.config.get("interval"),
                    "period": self.config.get("period"),
                    "data_source": self.config.get("data_source"),
                },
                config_signature_data={
                    "model": "long_term_rf_v1",
                    "fe_version": "v3",
                    "indicators_in_pipeline": True,
                    "outlier_clip_pct": (0.1, 99.9),
                    "outlier_clip_min_periods": 30,
                    "cv_gap": 50,
                    "ticker": self.config.get("ticker"),
                    "interval": self.config.get("interval"),
                    "period": self.config.get("period"),
                    "data_source": self.config.get("data_source"),
                },
                gap=50,
            )

            if cv_mae is not None:
                self.log_action(f"Cross-validated MAE (walk-forward): {cv_mae:.4f}", "info")

            self.log_action(f"Baseline MAE (predict zero return): {baseline_mae:.4f}", "info")
            if elasticnet_baseline_mae is not None:
                self.log_action(f"ElasticNet baseline MAE: {elasticnet_baseline_mae:.4f}", "info")

            # Predykcja na najnowszym wierszu (inference)
            try:
                check_is_fitted(pipeline)
            except NotFittedError as exc:
                self.log_action(f"Persisted pipeline not fitted or invalid: {exc}", "error")
                return

            predicted_return = float(pipeline.predict(inference_row)[0])
            self._log_feature_importance(pipeline, feature_columns)

        last_close = raw_data["Close"].iloc[-1]
        msg = f"Predicted 50-day return: {predicted_return:.4f} ({predicted_return*100:.2f}%), Last Close: {last_close:.2f}"

        # Use adaptive thresholds if enabled
        buy_threshold, sell_threshold = self._get_adaptive_thresholds(raw_data)

        if predicted_return >= buy_threshold:
            decision = "BUY"
            self.log_action(f"{msg} -> {decision} signal (positive return expected)", "info")
        elif predicted_return <= sell_threshold:
            decision = "SELL"
            self.log_action(
                f"{msg} -> {decision} signal (expected return below sell threshold {sell_threshold:.4f})", "info"
            )
        else:
            decision = "HOLD"
            self.log_action(
                f"{msg} -> {decision} signal (return {predicted_return:.4f} between thresholds "
                f"{sell_threshold:.4f} and {buy_threshold:.4f})",
                "warning",
            )

        trade_summary = self.order_executor.process_signal(
            asset, decision, raw_data["Close"].iloc[-1], self.risk_manager
        )
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")
        send_email(
            f"{decision} Signal for {asset} using {strategy_name}",
            f"{msg} -> {decision} signal | trade: {trade_summary}",
            self.config.get("notification_email"),
        )

    def _deduplicate_features(
        self, X: pd.DataFrame, threshold: float = 0.999
    ) -> tuple[pd.DataFrame, list[str]]:
        """Drop features that are almost perfectly correlated to reduce redundancy."""
        filled = X.copy()
        filled = filled.fillna(filled.median(numeric_only=True))
        corr = filled.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        if to_drop:
            X = X.drop(columns=to_drop)
        return X, to_drop

    def _indicator_set(self) -> set[str]:
        indicators = self.config.get("indicators", [])
        if not self.config.get("use_indicators", True):
            return set()
        if isinstance(indicators, str):
            indicators = [indicators]
        return {str(ind).lower() for ind in indicators}

    def _indicator_feature_columns(self, indicator_names: set[str]) -> list[str]:
        columns: list[str] = []
        if "sma" in indicator_names:
            columns.append("SMA")
        if "ema" in indicator_names:
            columns.append("EMA")
        if "macd" in indicator_names:
            columns.extend(["MACD", "Signal", "MACD_Histogram"])
        return columns

    def _build_outlier_clipper(self) -> ReturnOutlierClipper:
        return ReturnOutlierClipper(
            price_columns=["Close"],
            lower_pct=0.1,
            upper_pct=99.9,
            min_periods=30,
        )

    def _log_feature_importance(self, pipeline: Any, feature_columns: list[str]) -> None:
        """Log top feature importances for trained RandomForest."""
        try:
            model = pipeline.named_steps.get("model")
            if model is None or not hasattr(model, "feature_importances_"):
                return
            importances = model.feature_importances_
            if len(importances) != len(feature_columns):
                return
            pairs = sorted(zip(feature_columns, importances), key=lambda p: p[1], reverse=True)
            top = pairs[:10]
            self.log_action(f"Top feature importances: {top}", "info")
            zero = [name for name, imp in pairs if imp == 0]
            if zero:
                self.log_action(f"Zero-importance features (candidates for removal): {zero}", "warning")
        except (ValueError, TypeError, KeyError):
            return
