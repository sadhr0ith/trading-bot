from typing import Dict, List
import logging

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

from .strategy_base import StrategyBase
from indicators.macd import MACD
from indicators.rsi import RSI
from indicators.adx import ADX
from utils.email_notifications import send_email
from utils.model_persistence import ModelPersistence
from utils.strategy_helpers import train_or_load_pipeline
from utils.transformers import (
    FeatureSelector,
    IndicatorLagTransformer,
    LagFeatureTransformer,
    ReturnFeatureTransformer,
    RollingStatsTransformer,
    CalendarFeatureTransformer,
)

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

    def execute(self):
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]

        self.log_action(f"Executing short-term strategy for {asset} using XGBoost", "info")
        data = self.data.copy().sort_index()

        if data is None or data.empty:
            self.log_action("Input data frame is empty; skipping execution.", "warning")
            return

        data = self._apply_configured_indicators(data)
        data['target'] = data['Close'].pct_change(5).shift(-5)

        if len(data) < 50:
            self.log_action("Not enough data rows; skipping execution.", "warning")
            return

        feature_columns, indicator_cols = self._build_feature_columns(data)
        has_volume = 'Volume' in data.columns

        # Validate minimum data requirement for inference
        # Lag transformers need: max(lags)=10, Rolling needs: max(windows)=10
        # Safety margin: 25 rows minimum (20 for history + 5 buffer)
        MIN_INFERENCE_ROWS = 25
        if len(data) < MIN_INFERENCE_ROWS:
            self.log_action(
                f"Insufficient data for inference: need {MIN_INFERENCE_ROWS}+ rows, have {len(data)}. "
                f"Lag/rolling features require historical context.",
                "error"
            )
            return

        # Clean split: last MIN_INFERENCE_ROWS are out-of-sample (NOT in training)
        split_idx = len(data) - MIN_INFERENCE_ROWS
        data_train = data.iloc[:split_idx].copy()
        data_train = data_train[data_train['target'].notna()]
        data_inference = data.iloc[-MIN_INFERENCE_ROWS:].drop(columns=['target'])

        # Sanity check: ensure no overlap between train and inference
        assert len(data_train) + len(data_inference) <= len(data), "Train/inference overlap detected!"
        self.log_action(
            f"Training on {len(data_train)} rows, reserving last {MIN_INFERENCE_ROWS} for out-of-sample inference",
            "info"
        )

        if len(data_train) < 50:
            self.log_action("Not enough training data after removing missing targets.", "warning")
            return

        if len(data_train) < 80:
            self.log_action("Small dataset detected (<80 rows); using ElasticNet fallback for sanity.", "warning")
            return self._run_fallback_elasticnet(data_train, data_inference, feature_columns)

        X_train = data_train.drop(columns=['target'])
        y_train = data_train['target']
        baseline_mae = float(np.mean(np.abs(y_train)))

        persistence = ModelPersistence()

        def build_fe_pipeline():
            fe_steps = [
                ('lag_features', LagFeatureTransformer(columns=['Close'], lags=[1, 3, 5, 10])),
                ('return_features', ReturnFeatureTransformer(price_col='Close', periods=[1, 3, 5, 10])),
                ('rolling_stats', RollingStatsTransformer(
                    price_col='Close',
                    volume_col='Volume' if has_volume else None,
                    windows=[5, 10],
                )),
                ('calendar_features', CalendarFeatureTransformer()),
            ]

            if indicator_cols:
                fe_steps.append(('indicator_lags', IndicatorLagTransformer(
                    indicator_columns=indicator_cols,
                    lags=[1],
                    column_mapping=self._indicator_column_mapping(),
                )))

            fe_steps.append(('feature_selector', FeatureSelector(
                feature_columns=feature_columns,
                handle_missing='fill',
            )))

            return Pipeline(fe_steps)

        def build_model_pipeline(model):
            return Pipeline([
                ('feature_engineering', build_fe_pipeline()),
                ('scaler', StandardScaler()),
                ('model', model),
            ])

        def pipeline_factory():
            """Build complete ML pipeline: FE -> Scaler -> XGBoost."""
            return build_model_pipeline(XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                max_depth=6,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
            ))

        search_space: Dict[str, List] = {
            "model__max_depth": [4, 5, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__n_estimators": [200, 300, 500],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__min_child_weight": [1, 5, 10],
        }

        def tuner(pipeline, X, y, cv, scoring):
            # Light tuning for small datasets; full search otherwise
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
            best_pipeline = search.best_estimator_
            best_score = search.best_score_
            cv_metric = float(abs(best_score)) if "neg_mean_absolute_error" in scoring else float(best_score)
            tuner_meta = {
                "tuner": "RandomizedSearchCV",
                "best_params": search.best_params_,
                "n_iter": search.n_iter,
            }
            return best_pipeline, cv_metric, tuner_meta

        elasticnet_baseline_mae = None
        try:
            baseline_pipeline = build_model_pipeline(ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=5000,
                random_state=self.seed,
            ))
            baseline_tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train) // 50)))
            baseline_scores = cross_val_score(
                baseline_pipeline,
                X_train,
                y_train,
                cv=baseline_tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            elasticnet_baseline_mae = float(np.mean(np.abs(baseline_scores)))
        except Exception as exc:  # noqa: BLE001
            self.log_action(f"ElasticNet baseline evaluation failed: {exc}", "warning")

        pipeline, cv_mae = train_or_load_pipeline(
            key=strategy_name,
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
            },
            config_signature_data={
                "model": "short_term_xgb_v1",
                "search_space": search_space,
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

        # Conditional debug logging for NaN detection (only if logger is DEBUG level)
        if self.logger.isEnabledFor(logging.DEBUG):
            try:
                X_inference = pipeline.named_steps['feature_engineering'].transform(data_inference)
                if np.isnan(X_inference[-1]).any():
                    nan_indices = np.where(np.isnan(X_inference[-1]))[0]
                    self.logger.warning(f"NaN features detected in inference row: indices {nan_indices}")
            except Exception as e:  # noqa: BLE001
                self.logger.debug(f"Could not check for NaN features: {e}")

        # Predict on last 25 rows (for lag/rolling context), keep only last prediction
        predictions = pipeline.predict(data_inference)
        predicted_return = float(predictions[-1])  # Take last prediction (current row)
        self._log_feature_stats_for_inference(pipeline, data_inference)

        last_macd = data['MACD'].iloc[-1] if 'MACD' in data.columns else None
        last_signal = data['Signal'].iloc[-1] if 'Signal' in data.columns else None

        hold_threshold = 0.005
        msg = f"Predicted 5-day return: {predicted_return:.4f}"
        if last_macd is not None and last_signal is not None:
            msg += f", MACD: {last_macd:.4f}, Signal: {last_signal:.4f}"

        if predicted_return > hold_threshold and (last_macd is None or last_macd > last_signal):
            decision = "BUY"
            reason = "positive expected return with MACD confirmation" if last_macd is not None else "positive expected return"
        elif predicted_return < -hold_threshold and (last_macd is None or last_macd < last_signal):
            decision = "SELL"
            reason = "negative expected return with MACD confirmation" if last_macd is not None else "negative expected return"
        else:
            decision = "HOLD"
            reason = "signal below threshold or indicators not aligned"

        self.log_action(f"{msg} -> {decision} ({reason})", "info" if decision != "HOLD" else "warning")
        trade_summary = self.order_executor.process_signal(asset, decision, data['Close'].iloc[-1], self.risk_manager)
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")

        send_email(
            f"{decision} Signal for {asset} using {strategy_name}",
            f"{msg} -> {decision} ({reason}) | trade: {trade_summary}",
            self.config['notification_email'],
        )

    def _apply_configured_indicators(self, data):
        """Attach configured indicators to the price frame."""
        if self.config.get("use_indicators", True) and "macd" in self.config["indicators"]:
            self.log_action("Calculating MACD indicator...", "info")
            macd_indicator = MACD(data)
            data = data.drop(columns=['MACD', 'Signal', 'MACD_Histogram'], errors='ignore')
            macd_data = macd_indicator.calculate()
            data = data.join(macd_data[['MACD', 'Signal', 'MACD_Histogram']])

        if self.config.get("use_indicators", True) and "rsi" in self.config["indicators"]:
            self.log_action("Calculating RSI indicator...", "info")
            rsi_indicator = RSI(data)
            data['RSI'] = rsi_indicator.calculate()
        if self.config.get("use_indicators", True) and "adx" in self.config["indicators"]:
            self.log_action("Calculating ADX indicator...", "info")
            adx_indicator = ADX(data)
            data = data.drop(columns=['ADX', 'Plus_DI', 'Minus_DI'], errors='ignore')
            adx_data = adx_indicator.calculate()
            data = data.join(adx_data[['ADX', 'Plus_DI', 'Minus_DI']])

        return data

    def _build_feature_columns(self, data):
        """Define the feature set based on available columns."""
        feature_columns: List[str] = [
            'Close_lag_1', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10',
            'Return_lag_1', 'Return_lag_3', 'Return_lag_5', 'Return_lag_10',
            'SMA_5', 'SMA_10', 'Volatility_5', 'Volatility_10',
            'day_of_week', 'month', 'is_month_start', 'is_month_end',
        ]

        has_volume = 'Volume' in data.columns
        if has_volume:
            feature_columns.extend(['Volume_MA_5', 'Volume_MA_10'])

        indicator_cols: List[str] = []
        if 'MACD' in data.columns:
            indicator_cols.extend(['MACD', 'Signal'])
            feature_columns.extend(['MACD_lag_1', 'Signal_lag_1'])
        if 'RSI' in data.columns:
            indicator_cols.append('RSI')
            feature_columns.append('RSI_lag_1')
        if 'ADX' in data.columns:
            indicator_cols.extend(['ADX', 'Plus_DI', 'Minus_DI'])
            feature_columns.extend(['ADX_lag_1', 'Plus_DI_lag_1', 'Minus_DI_lag_1'])

        return feature_columns, indicator_cols

    def _indicator_column_mapping(self):
        return {
            'DI+': 'Plus_DI',
            'DI-': 'Minus_DI',
            '+DI': 'Plus_DI',
            '-DI': 'Minus_DI',
        }

    def _log_feature_stats_for_inference(self, pipeline, raw_df):
        """Log summary stats for the transformed inference row to spot drift/NaN."""
        try:
            fe = pipeline.named_steps.get('feature_engineering')
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
        except Exception:  # noqa: BLE001
            return

    def _run_fallback_elasticnet(self, data_train, data_inference, feature_columns):
        """Fallback path for tiny datasets: use ElasticNet only."""
        X_train = data_train.drop(columns=['target'])
        y_train = data_train['target']
        fe_steps = [
            ('lag_features', LagFeatureTransformer(columns=['Close'], lags=[1, 3, 5, 10])),
            ('return_features', ReturnFeatureTransformer(price_col='Close', periods=[1, 3, 5, 10])),
            ('rolling_stats', RollingStatsTransformer(
                price_col='Close',
                volume_col='Volume' if 'Volume' in X_train.columns else None,
                windows=[5, 10],
            )),
            ('calendar_features', CalendarFeatureTransformer()),
            ('feature_selector', FeatureSelector(feature_columns=feature_columns, handle_missing='fill')),
        ]
        pipeline = Pipeline([
            ('feature_engineering', Pipeline(fe_steps)),
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=self.seed)),
        ])
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
        trade_summary = self.order_executor.process_signal(self.config["ticker"], decision, data_train['Close'].iloc[-1], self.risk_manager)
        self.log_action(f"Fallback trade summary: {trade_summary}", "info")
        send_email(
            f"{decision} Signal (fallback ElasticNet) for {self.config['ticker']} using {self.config['strategy']}",
            f"{predicted_return:.4f} -> {decision} ({reason}) | trade: {trade_summary}",
            self.config['notification_email'],
        )
        return decision
