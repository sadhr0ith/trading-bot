# strategies/mid_term_strategy.py
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import HalvingRandomSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

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


class MidTermStrategy(StrategyBase):
    """Mid-term (20-day) RandomForest on lag/return/vol/drawdown features with indicator add-ons and correlation pruning."""

    MIN_ROWS: int = 50

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _run_strategy(self, data: pd.DataFrame) -> None:
        asset = self.config.get("ticker", "UNKNOWN")
        inference_only = self.config.get("inference_only", False)
        persistence_key = build_persistence_key(
            strategy=self.config.get("strategy", "mid_term"),
            data_source=self.config.get("data_source"),
            ticker=asset,
            interval=self.config.get("interval"),
        )
        self.log_action("Executing mid-term trading strategy with Random Forest", "info")
        raw_data = data
        indicator_names = self._indicator_set()

        lags = [5, 10, 20, 60, 120]
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
            ],
        )

        features = fe_pipeline.transform(data)
        target_frame = create_forward_return_target(clipper.transform(data), horizon=20)
        target = target_frame["target"]

        clean_rows = features.dropna()
        if clean_rows.empty:
            self.log_action("Not enough complete rows to generate inference features.", "warning")
            return
        inference_row = clean_rows.tail(1)
        inference_index = inference_row.index[0]

        combined = features.join(target.rename("target")).dropna()
        if combined.empty:
            self.log_action("Not enough data after feature engineering; skipping execution.", "warning")
            return
        train_mask = combined.index != inference_index
        X = combined.loc[train_mask, feature_columns]
        y = combined.loc[train_mask, "target"]
        inference_row = inference_row[feature_columns]

        if not inference_only:
            # Remove highly correlated / duplicate features to reduce noise
            X, dropped_cols = self._deduplicate_features(X)
            if dropped_cols:
                self.log_action(f"Removed highly correlated features (r>0.999): {sorted(dropped_cols)}", "info")
            feature_columns = list(X.columns)
            inference_row = inference_row[feature_columns]

        if len(X) < 40 and not inference_only:
            self.log_action("Not enough data after feature engineering; skipping execution.", "warning")
            return
        baseline_mae = float(np.mean(np.abs(y))) if not y.empty else 0.0

        persistence = ModelPersistence()
        preprocess = ColumnTransformer([("num", StandardScaler(), feature_columns)], remainder="drop")

        def build_model_pipeline(model):
            return Pipeline(
                steps=[
                    ("preprocess", preprocess),
                    ("model", model),
                ],
            )

        pipeline_factory = lambda: build_model_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                random_state=self.seed,
                n_jobs=-1,
            ),
        )

        param_distributions = {
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [6, 8, 10, 12],
            "model__min_samples_leaf": [1, 2, 4],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["auto", "sqrt"],
        }

        def tuner(pipeline, X_train, y_train, cv, scoring):
            if len(X_train) < 120:
                self.log_action("Dataset small; skipping HalvingRandomSearchCV and using default RF params.", "warning")
                pipeline.fit(X_train, y_train)
                return pipeline, None, {}
            search = HalvingRandomSearchCV(
                pipeline,
                param_distributions=param_distributions,
                factor=2,
                random_state=self.seed,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_score = search.best_score_
            cv_metric = float(abs(best_score)) if "neg_mean_absolute_error" in scoring else float(best_score)
            tuner_meta = {
                "tuner": "HalvingRandomSearchCV",
                "best_params": search.best_params_,
                "n_candidates": search.n_candidates_,
                "search_space": list(param_distributions.keys()),
            }
            return best_pipeline, cv_metric, tuner_meta

        elasticnet_baseline_mae = None
        if not inference_only:
            try:
                enet_pipeline = build_model_pipeline(
                    ElasticNet(
                        alpha=0.01,
                        l1_ratio=0.5,
                        max_iter=5000,
                        random_state=self.seed,
                    )
                )
                baseline_tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 40)), gap=20)
                baseline_scores = cross_val_score(
                    enet_pipeline,
                    X,
                    y,
                    cv=baseline_tscv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                elasticnet_baseline_mae = float(np.mean(np.abs(baseline_scores)))
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                self.log_action(f"ElasticNet baseline evaluation failed: {exc}", "warning")

        if inference_only:
            try:
                pipeline, _ = train_or_load_pipeline(
                    key=persistence_key,
                    persistence=persistence,
                    pipeline_factory=pipeline_factory,  # unused in load_only
                    X=X,
                    y=y,
                    feature_columns=feature_columns,
                    load_only=True,
                    config_signature_data={
                        "model": "mid_term_rf_v1",
                        "search_space": param_distributions,
                        "fe_version": "v3",
                        "indicators_in_pipeline": True,
                        "outlier_clip_pct": (0.1, 99.9),
                        "outlier_clip_min_periods": 30,
                        "cv_gap": 20,
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
            last_close = raw_data["Close"].iloc[-1]
            msg = f"Predicted 20-day return (inference-only): {predicted_return:.4f}, last close: {last_close:.2f}"
            # Use adaptive thresholds if enabled
            buy_threshold, sell_threshold = self._get_adaptive_thresholds(raw_data)
            if predicted_return > buy_threshold:
                decision = "BUY"
                reason = f"expected return above {buy_threshold*100:.1f}%"
            elif predicted_return < sell_threshold:
                decision = "SELL"
                reason = f"expected return below {sell_threshold*100:.1f}%"
            else:
                decision = "HOLD"
                reason = "expected return within threshold"
            self.log_action(f"{msg} -> {decision} ({reason})", "info" if decision != "HOLD" else "warning")
            trade_summary = self.order_executor.process_signal(
                asset, decision, raw_data["Close"].iloc[-1], self.risk_manager
            )
            if trade_summary.get("status") not in {"noop", "already_long"}:
                self.log_action(f"Paper trade summary: {trade_summary}", "info")
            return

        pipeline, cv_mae = train_or_load_pipeline(
            key=persistence_key,
            persistence=persistence,
            pipeline_factory=pipeline_factory,
            X=X,
            y=y,
            feature_columns=feature_columns,
            split_divisor=40,
            metadata={
                "baseline_mae": baseline_mae,
                "baseline_elasticnet_mae": elasticnet_baseline_mae,
                "train_rows": len(y),
                "ticker": self.config.get("ticker"),
                "interval": self.config.get("interval"),
                "period": self.config.get("period"),
                "data_source": self.config.get("data_source"),
            },
            config_signature_data={
                "model": "mid_term_rf_v1",
                "search_space": param_distributions,
                "fe_version": "v3",
                "indicators_in_pipeline": True,
                "outlier_clip_pct": (0.1, 99.9),
                "outlier_clip_min_periods": 30,
                "cv_gap": 20,
                "ticker": self.config.get("ticker"),
                "interval": self.config.get("interval"),
                "period": self.config.get("period"),
                "data_source": self.config.get("data_source"),
            },
            gap=20,
            tuner=tuner,
        )

        if cv_mae is not None:
            self.log_action(f"Cross-validated MAE (walk-forward): {cv_mae:.4f}", "info")

        self.log_action(f"Baseline MAE (predict 0 return): {baseline_mae:.4f}", "info")
        if elasticnet_baseline_mae is not None:
            self.log_action(f"ElasticNet baseline MAE: {elasticnet_baseline_mae:.4f}", "info")

        # Use pipeline directly from train_or_load_pipeline (no reload needed)
        try:
            check_is_fitted(pipeline)
        except NotFittedError as exc:
            self.log_action(f"Pipeline not fitted or invalid: {exc}", "error")
            return
        predicted_return = float(pipeline.predict(inference_row)[0])
        self._log_feature_importance(pipeline, feature_columns)
        last_close = raw_data["Close"].iloc[-1]
        msg = f"Predicted 20-day return: {predicted_return:.4f}, last close: {last_close:.2f}"

        # Use adaptive thresholds if enabled
        buy_threshold, sell_threshold = self._get_adaptive_thresholds(raw_data)

        if predicted_return > buy_threshold:
            decision = "BUY"
            reason = f"expected return above {buy_threshold*100:.1f}%"
        elif predicted_return < sell_threshold:
            decision = "SELL"
            reason = f"expected return below {sell_threshold*100:.1f}%"
        else:
            decision = "HOLD"
            reason = "expected return within threshold"

        self.log_action(f"{msg} -> {decision} ({reason})", "info" if decision != "HOLD" else "warning")
        trade_summary = self.order_executor.process_signal(
            asset, decision, raw_data["Close"].iloc[-1], self.risk_manager
        )
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")

    def _indicator_set(self) -> set[str]:
        indicators = self.config.get("indicators", [])
        if not self.config.get("use_indicators", True):
            return set()
        if isinstance(indicators, str):
            indicators = [indicators]
        return {str(ind).lower() for ind in indicators}

    def _indicator_feature_columns(self, indicator_names: set[str]) -> list[str]:
        columns: list[str] = []
        if "macd" in indicator_names:
            columns.extend(["MACD", "MACD_Histogram"])
        if "bollinger_bands" in indicator_names:
            columns.extend(["BB_Width", "BB_Middle"])
        return columns

    def _build_outlier_clipper(self) -> ReturnOutlierClipper:
        return ReturnOutlierClipper(
            price_columns=["Close"],
            lower_pct=0.1,
            upper_pct=99.9,
            min_periods=30,
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
