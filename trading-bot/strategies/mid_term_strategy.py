# strategies/mid_term_strategy.py

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import HalvingRandomSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from indicators.bollinger_bands import BollingerBands
from indicators.macd import MACD
from utils.model_persistence import ModelPersistence
from utils.strategy_helpers import train_or_load_pipeline

from .strategy_base import StrategyBase


class MidTermStrategy(StrategyBase):
    """Mid-term (20-day) RandomForest on lag/return/vol/drawdown features with indicator add-ons and correlation pruning."""

    MIN_ROWS = 50

    def _add_indicators(self, data):
        out = data.copy()
        if self.config.get("use_indicators", True) and "macd" in self.config.get("indicators", []):
            self.log_action("Calculating MACD indicator...", "info")
            macd_indicator = MACD(out)
            out = out.drop(columns=["MACD", "Signal", "MACD_Histogram"], errors="ignore")
            out = out.join(macd_indicator.calculate())

        if self.config.get("use_indicators", True) and "bollinger_bands" in self.config.get("indicators", []):
            self.log_action("Calculating Bollinger Bands indicator...", "info")
            bb_indicator = BollingerBands(out)
            out = out.drop(columns=["BB_Middle", "BB_Upper", "BB_Lower", "BB_Width"], errors="ignore")
            out = out.join(bb_indicator.calculate())
        return out

    def _run_strategy(self, data):
        asset = self.config.get("ticker", "UNKNOWN")
        self.log_action("Executing mid-term trading strategy with Random Forest", "info")

        for lag in [5, 10, 20, 60, 120]:
            data[f"Close_lag_{lag}"] = data["Close"].shift(lag)
            data[f"Return_lag_{lag}"] = data["Close"].pct_change(lag)
            data[f"Volatility_{lag}"] = data["Close"].pct_change().rolling(lag).std()
            rolling_max = data["Close"].rolling(lag).max()
            data[f"Drawdown_{lag}"] = (data["Close"] / rolling_max) - 1

        data["target"] = data["Close"].pct_change(20).shift(-20)

        feature_columns: list[str] = [
            "Close_lag_5",
            "Close_lag_10",
            "Close_lag_20",
            "Close_lag_60",
            "Close_lag_120",
            "Return_lag_5",
            "Return_lag_10",
            "Return_lag_20",
            "Return_lag_60",
            "Return_lag_120",
            "Volatility_5",
            "Volatility_10",
            "Volatility_20",
            "Volatility_60",
            "Volatility_120",
            "Drawdown_5",
            "Drawdown_10",
            "Drawdown_20",
            "Drawdown_60",
            "Drawdown_120",
        ]
        for col in ["MACD", "MACD_Histogram", "BB_Width", "BB_Middle"]:
            if col in data.columns:
                feature_columns.append(col)

        features = data[feature_columns]
        target = data["target"]

        inference_row = features.tail(1)
        if inference_row.isna().any().any():
            clean_rows = features.dropna()
            if clean_rows.empty:
                self.log_action("Not enough complete rows to generate inference features.", "warning")
                return
            inference_row = clean_rows.tail(1)
        inference_index = inference_row.index[0]

        train_mask = (features.index != inference_index) & target.notna()
        X = features.loc[train_mask].dropna()
        y = target.loc[X.index]

        # Remove highly correlated / duplicate features to reduce noise
        X, dropped_cols = self._deduplicate_features(X)
        if dropped_cols:
            self.log_action(f"Removed highly correlated features (r>0.999): {sorted(dropped_cols)}", "info")
        feature_columns = list(X.columns)
        inference_row = inference_row[feature_columns]

        if len(X) < 40:
            self.log_action("Not enough data after feature engineering; skipping execution.", "warning")
            return
        baseline_mae = float(np.mean(np.abs(y)))

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
        try:
            enet_pipeline = build_model_pipeline(
                ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.5,
                    max_iter=5000,
                    random_state=self.seed,
                )
            )
            baseline_tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 40)))
            baseline_scores = cross_val_score(
                enet_pipeline,
                X,
                y,
                cv=baseline_tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            elasticnet_baseline_mae = float(np.mean(np.abs(baseline_scores)))
        except Exception as exc:  # noqa: BLE001
            self.log_action(f"ElasticNet baseline evaluation failed: {exc}", "warning")

        pipeline, cv_mae = train_or_load_pipeline(
            key=self.config.get("strategy", "mid_term"),
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
            },
            config_signature_data={
                "model": "mid_term_rf_v1",
                "search_space": param_distributions,
                "fe_version": "v1",
            },
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
        except Exception as exc:  # noqa: BLE001
            self.log_action(f"Pipeline not fitted or invalid: {exc}", "error")
            return
        predicted_return = float(pipeline.predict(inference_row)[0])
        self._log_feature_importance(pipeline, feature_columns)
        last_close = data["Close"].iloc[-1]
        msg = f"Predicted 20-day return: {predicted_return:.4f}, last close: {last_close:.2f}"

        # Get thresholds from config (default: 2% = 0.02)
        buy_threshold = self.config.get("buy_threshold", 0.02)
        sell_threshold = self.config.get("sell_threshold", -0.02)

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
        trade_summary = self.order_executor.process_signal(asset, decision, data["Close"].iloc[-1], self.risk_manager)
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")

    def _deduplicate_features(self, X, threshold: float = 0.999):
        """Drop features that are almost perfectly correlated to reduce redundancy."""
        corr = X.fillna(0).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        if to_drop:
            X = X.drop(columns=to_drop)
        return X, to_drop

    def _log_feature_importance(self, pipeline, feature_columns):
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
        except Exception:  # noqa: BLE001
            return
