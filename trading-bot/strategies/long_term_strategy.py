
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from indicators.ema import EMA
from indicators.macd import MACD
from indicators.sma import SMA
from utils.email_notifications import send_email
from utils.feature_engineering import (
    add_indicator_columns,
    add_lag_features,
    build_lag_feature_columns,
    create_forward_return_target,
)
from utils.model_persistence import ModelPersistence
from utils.strategy_helpers import train_or_load_pipeline

from .strategy_base import StrategyBase


class LongTermStrategy(StrategyBase):
    """Long-term (50-day) RandomForest using long lags, return/vol/drawdown features, indicators, and correlation pruning."""
    MIN_ROWS = 50

    def _run_strategy(self, data):
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]

        self.log_action(f"Executing long-term strategy for {asset} using Random Forest and SMA/EMA", "info")

        if self.config.get("use_indicators", True) and "sma" in self.config["indicators"]:
            self.log_action("Calculating SMA indicator...", "info")
            sma_indicator = SMA(data, window=200, alias="SMA")
            data = data.drop(columns=["SMA"], errors="ignore")
            data = data.join(sma_indicator.calculate())

        if self.config.get("use_indicators", True) and "ema" in self.config["indicators"]:
            self.log_action("Calculating EMA indicator...", "info")
            ema_indicator = EMA(data, span=50, alias="EMA")
            data = data.drop(columns=["EMA"], errors="ignore")
            data = data.join(ema_indicator.calculate())

        if self.config.get("use_indicators", True) and "macd" in self.config["indicators"]:
            self.log_action("Calculating MACD indicator...", "info")
            macd_indicator = MACD(data)
            data = data.drop(columns=["MACD", "Signal", "MACD_Histogram"], errors="ignore")
            data = data.join(macd_indicator.calculate())

        # Add lag-based features using utility function
        lags = [20, 60, 120, 250]
        data = add_lag_features(data, lags)

        # Create forward-looking target (50-day return)
        data = create_forward_return_target(data, horizon=50)

        # Build feature column list from lags (avoid using current Close - only lags and indicators)
        feature_columns = build_lag_feature_columns(lags)

        # Add indicator columns if they exist
        feature_columns = add_indicator_columns(feature_columns, data, ["SMA", "EMA", "MACD"])

        # Dropna dla features I target
        combined = data[feature_columns + ["target"]].dropna()
        if combined.empty or len(combined) < 50:
            self.log_action("Not enough data after feature engineering; skipping execution.", "warning")
            return

        features = combined[feature_columns]
        y_aligned = combined["target"]

        # Remove highly correlated / duplicate features to reduce redundancy
        features, dropped_cols = self._deduplicate_features(features)
        if dropped_cols:
            self.log_action(f"Removed highly correlated features (r>0.999): {sorted(dropped_cols)}", "info")
        feature_columns = list(features.columns)
        baseline_mae = float(np.mean(np.abs(y_aligned)))
        elasticnet_baseline_mae = None
        try:
            enet_pipeline = Pipeline(
                steps=[
                    ("preprocess", ColumnTransformer([("num", StandardScaler(), feature_columns)], remainder="drop")),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=self.seed)),
                ],
            )
            baseline_tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(features) // 60)))
            baseline_scores = cross_val_score(
                enet_pipeline,
                features,
                y_aligned,
                cv=baseline_tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            elasticnet_baseline_mae = float(np.mean(np.abs(baseline_scores)))
        except Exception as exc:  # noqa: BLE001
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

        # Train or load pipeline (unified persistence)
        pipeline, cv_mae = train_or_load_pipeline(
            key=strategy_name,
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
            },
            config_signature_data={
                "model": "long_term_rf_v1",
                "fe_version": "v1",
            },
        )

        if cv_mae is not None:
            self.log_action(f"Cross-validated MAE (walk-forward): {cv_mae:.4f}", "info")

        self.log_action(f"Baseline MAE (predict zero return): {baseline_mae:.4f}", "info")
        if elasticnet_baseline_mae is not None:
            self.log_action(f"ElasticNet baseline MAE: {elasticnet_baseline_mae:.4f}", "info")

        # Predykcja na najnowszym wierszu (inference)
        latest_features = features.tail(1)
        try:
            check_is_fitted(pipeline)
        except Exception as exc:  # noqa: BLE001
            self.log_action(f"Persisted pipeline not fitted or invalid: {exc}", "error")
            return

        predicted_return = float(pipeline.predict(latest_features)[0])
        self._log_feature_importance(pipeline, feature_columns)

        last_close = data["Close"].iloc[-1]
        msg = f"Predicted 50-day return: {predicted_return:.4f} ({predicted_return*100:.2f}%), Last Close: {last_close:.2f}"

        # Get thresholds from config (default: 0.5% = 0.005)
        buy_threshold = self.config.get("buy_threshold", 0.005)
        sell_threshold = self.config.get("sell_threshold", -0.005)

        if abs(predicted_return) < buy_threshold:
            decision = "HOLD"
            self.log_action(
                f"{msg} -> {decision} signal (predicted return too small: {abs(predicted_return):.4f} < {buy_threshold})",
                "warning",
            )
        elif predicted_return > buy_threshold:
            decision = "BUY"
            self.log_action(f"{msg} -> {decision} signal (positive return expected)", "info")
        else:  # predicted_return < sell_threshold
            decision = "SELL"
            self.log_action(f"{msg} -> {decision} signal (negative return expected)", "info")

        trade_summary = self.order_executor.process_signal(asset, decision, data["Close"].iloc[-1], self.risk_manager)
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")
        send_email(
            f"{decision} Signal for {asset} using {strategy_name}",
            f"{msg} -> {decision} signal | trade: {trade_summary}",
            self.config["notification_email"],
        )

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
