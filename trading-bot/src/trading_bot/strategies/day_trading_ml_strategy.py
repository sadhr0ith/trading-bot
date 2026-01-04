from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

try:
    from xgboost import XGBRegressor

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None
    _XGB_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingRegressor

from trading_bot.backtest.adapters import compute_atr
from trading_bot.core.exceptions import InsufficientDataError
from trading_bot.indicators.adx import ADX
from trading_bot.indicators.bollinger_bands import BollingerBands
from trading_bot.indicators.rsi import RSI
from trading_bot.models.signal import SignalAction
from trading_bot.strategies.strategy_base import StrategyBase
from trading_bot.utils.email_notifications import send_email
from trading_bot.utils.feature_engineering import (
    add_lag_features,
    build_lag_feature_columns,
    create_forward_return_target,
)
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.proxy_metrics import long_only_proxy
from trading_bot.utils.strategy_helpers import _build_config_signature, build_persistence_key, train_or_load_pipeline


class DayTradingMLStrategy(StrategyBase):
    """Tabular ML strategy for intraday return prediction."""

    MIN_ROWS = 200

    def __init__(self, config, data: pd.DataFrame) -> None:
        super().__init__(config, data)
        self.horizon = int(config.get("return_horizon", 1))
        self.prediction_threshold = float(config.get("prediction_threshold", 0.0))
        self.min_inference_rows = int(config.get("min_inference_rows", 120))
        self.slippage_rate = float(config.get("slippage_rate", 0.0002))
        self.min_improvement = float(config.get("ml_min_improvement", 0.05))
        self.max_drawdown = config.get("ml_max_drawdown")
        if self.max_drawdown is not None:
            self.max_drawdown = float(self.max_drawdown)

    def _build_features(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        out = data.copy()
        if "Volume" not in out.columns:
            out["Volume"] = 0.0

        out = create_forward_return_target(out, horizon=self.horizon)
        lags = [1, 2, 3, 5, 10]
        out = add_lag_features(out, lags=lags, price_col="Close")
        out["ATR"] = compute_atr(out, window=14)
        out["RSI"] = RSI(out, period=14).calculate()

        bb = BollingerBands(out, window=20, num_std=2.0).calculate()
        out["BB_Width"] = bb["BB_Width"]

        adx = ADX(out, window=14).calculate()
        out["ADX"] = adx["ADX"]

        out["Volatility_10"] = out["Close"].pct_change().rolling(10).std()

        feature_columns = build_lag_feature_columns(lags)
        feature_columns.extend(["ATR", "RSI", "BB_Width", "ADX", "Volatility_10", "Volume"])

        out = out.dropna(subset=feature_columns + ["target"])
        return out, feature_columns

    def _pipeline_factory(self) -> Pipeline:
        if _XGB_AVAILABLE:
            model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.seed,
            )
        else:
            model = HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=300,
                random_state=self.seed,
            )
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    def _baseline_mae(self, y: pd.Series) -> tuple[float, float | None]:
        baseline_zero = float(np.mean(np.abs(y)))
        baseline_last = y.shift(1).dropna()
        if baseline_last.empty:
            return baseline_zero, None
        aligned = y.loc[baseline_last.index]
        baseline_last_mae = float(np.mean(np.abs(aligned - baseline_last)))
        return baseline_zero, baseline_last_mae

    def _proxy_metrics(self, y_true: pd.Series, y_pred: np.ndarray, threshold: float) -> dict[str, float | None]:
        if len(y_true) == 0:
            return {"hit_rate": None, "pnl_proxy": None, "max_drawdown": None}
        mask = np.abs(y_pred) > threshold
        if not np.any(mask):
            return {"hit_rate": None, "pnl_proxy": None, "max_drawdown": None}
        y_true_masked = y_true.values[mask]
        y_pred_masked = y_pred[mask]
        hits = np.sign(y_true_masked) == np.sign(y_pred_masked)
        hit_rate = float(np.mean(hits)) if hits.size else None

        long_returns = np.where(y_pred_masked > 0, y_true_masked, 0.0)
        equity = pd.Series((1.0 + long_returns).cumprod())
        max_dd = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else None
        pnl_proxy = float(equity.iloc[-1] - 1.0) if not equity.empty else None

        return {"hit_rate": hit_rate, "pnl_proxy": pnl_proxy, "max_drawdown": max_dd}

    def _quality_gate(self, cv_mae: float | None, baseline_mae: float | None, proxy_max_dd: float | None) -> bool:
        if cv_mae is None or baseline_mae is None:
            return True
        if baseline_mae <= 0:
            return True
        if cv_mae > baseline_mae * (1.0 - self.min_improvement):
            self.log_action(
                f"Quality gate failed: CV MAE {cv_mae:.6f} not better than baseline {baseline_mae:.6f}",
                "warning",
            )
            return False
        if self.max_drawdown is not None and proxy_max_dd is not None:
            if abs(proxy_max_dd) > self.max_drawdown:
                self.log_action(
                    f"Quality gate failed: max drawdown {proxy_max_dd:.4f} exceeds {self.max_drawdown:.4f}",
                    "warning",
                )
                return False
        return True

    def _gate_from_metadata(self, meta: dict) -> bool:
        cv_mae = meta.get("cv_mae", meta.get("mae_cv"))
        baseline_mae = meta.get("baseline_mae")
        proxy_max_dd = meta.get("max_drawdown_proxy_net", meta.get("max_drawdown_proxy"))

        if not self._quality_gate(cv_mae, baseline_mae, proxy_max_dd):
            return False

        min_hit_rate = self.config.get("ml_min_hit_rate")
        hit_rate = meta.get("hit_rate")
        if min_hit_rate is not None and hit_rate is not None and hit_rate < min_hit_rate:
            self.log_action(
                f"Quality gate failed: hit rate {hit_rate:.2%} below {min_hit_rate:.2%}.",
                "warning",
            )
            return False

        pnl_proxy = meta.get("pnl_proxy_net", meta.get("pnl_proxy"))
        if pnl_proxy is not None and pnl_proxy <= 0:
            self.log_action("Quality gate failed: PnL proxy <= 0.", "warning")
            return False

        return True

    def _min_edge(self) -> float:
        fee = float(self.risk_manager.trading_fee)
        base = abs(self.prediction_threshold)
        return base + fee + self.slippage_rate

    def _decide_signal(self, predicted_return: float) -> str:
        min_edge = self._min_edge()
        if predicted_return > min_edge:
            return "BUY"
        if predicted_return < -min_edge:
            return "SELL"
        return "HOLD"

    def _run_strategy(self, data: pd.DataFrame) -> None:
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]
        inference_only = self.config.get("inference_only", False)

        feature_data, feature_columns = self._build_features(data)
        if len(feature_data) < self.min_inference_rows:
            raise InsufficientDataError(
                f"Need at least {self.min_inference_rows} rows for inference, got {len(feature_data)}"
            )

        data_inference = feature_data.iloc[-self.min_inference_rows :]
        data_train = feature_data.iloc[: -self.min_inference_rows]

        if data_train.empty and not inference_only:
            raise InsufficientDataError("Not enough training data after feature engineering.")

        X_inference = data_inference[feature_columns]
        persistence_key = build_persistence_key(
            strategy=strategy_name,
            data_source=self.config.get("data_source"),
            ticker=asset,
            interval=self.config.get("interval"),
        )

        persistence = ModelPersistence()

        if inference_only:
            artifact = persistence.load(persistence_key)
            if not artifact:
                self.log_action("Inference-only mode but no persisted model found.", "warning")
                return
            pipeline = artifact.get("model")
            meta = artifact.get("metadata", {}) if artifact else {}
            trade_enabled = self._gate_from_metadata(meta or {})
            if pipeline is None:
                self.log_action("Inference-only mode but model artifact missing; skipping.", "warning")
                return
            self._run_inference(pipeline, X_inference, asset, strategy_name, trade_enabled=trade_enabled)
            return

        X_train = data_train[feature_columns]
        y_train = data_train["target"]

        baseline_zero, baseline_last = self._baseline_mae(y_train)
        baseline_mae = min(
            [val for val in [baseline_zero, baseline_last] if val is not None],
            default=baseline_zero,
        )

        config_blob = self.config.model_dump() if hasattr(self.config, "model_dump") else self.config
        config_signature_payload = {
            "model": "day_trading_ml_v1",
            "horizon": self.horizon,
            "features": feature_columns,
            "config": config_blob,
        }
        config_signature = _build_config_signature(config_signature_payload)

        pipeline, cv_mae = train_or_load_pipeline(
            key=persistence_key,
            persistence=persistence,
            pipeline_factory=self._pipeline_factory,
            X=X_train,
            y=y_train,
            feature_columns=feature_columns,
            gap=self.horizon,
            metadata={
                "baseline_mae_zero": baseline_zero,
                "baseline_mae_last": baseline_last,
                "baseline_mae": baseline_mae,
                "train_rows": len(y_train),
                "horizon": self.horizon,
                "config_signature": config_signature,
                "ticker": asset,
                "interval": self.config.get("interval"),
                "period": self.config.get("period"),
                "data_source": self.config.get("data_source"),
            },
            config_signature_data=config_signature_payload,
        )

        try:
            check_is_fitted(pipeline)
        except NotFittedError as exc:
            self.log_action(f"Pipeline not fitted: {exc}", "error")
            return

        pred_train = pipeline.predict(X_train)
        threshold = self._min_edge()
        proxy = self._proxy_metrics(y_train, pred_train, threshold)
        cost_per_side = float(self.risk_manager.trading_fee) + float(self.slippage_rate)
        proxy_net = long_only_proxy(y_train.values, pred_train, threshold=threshold, cost_per_side=cost_per_side)
        trade_enabled = self._quality_gate(cv_mae, baseline_mae, proxy_net.get("max_drawdown"))

        latest_idx = X_train.index.max() if hasattr(X_train, "index") else None
        persistence.save(
            persistence_key,
            pipeline,
            scaler=None,
            metadata={
                "baseline_mae_zero": baseline_zero,
                "baseline_mae_last": baseline_last,
                "baseline_mae": baseline_mae,
                "cv_mae": cv_mae,
                "hit_rate": proxy.get("hit_rate"),
                "pnl_proxy": proxy.get("pnl_proxy"),
                "max_drawdown_proxy": proxy.get("max_drawdown"),
                "pnl_proxy_net": proxy_net.get("pnl_proxy"),
                "max_drawdown_proxy_net": proxy_net.get("max_drawdown"),
                "proxy_entries": proxy_net.get("entries"),
                "proxy_exits": proxy_net.get("exits"),
                "config_signature": config_signature,
                "trained_until": str(latest_idx),
                "feature_columns": feature_columns,
                "horizon": self.horizon,
                "fe_version": "v1",
            },
        )

        self.log_action(f"Baseline MAE (zero): {baseline_zero:.6f}", "info")
        if baseline_last is not None:
            self.log_action(f"Baseline MAE (last return): {baseline_last:.6f}", "info")
        if cv_mae is not None:
            self.log_action(f"CV MAE: {cv_mae:.6f}", "info")
        if proxy.get("hit_rate") is not None:
            self.log_action(f"Hit rate @ threshold: {proxy['hit_rate']:.2%}", "info")

        self._run_inference(pipeline, X_inference, asset, strategy_name, trade_enabled=trade_enabled)

    def _run_inference(self, pipeline, X_inference: pd.DataFrame, asset: str, strategy_name: str, trade_enabled: bool):
        prediction = float(pipeline.predict(X_inference)[-1])
        decision = self._decide_signal(prediction)
        if not trade_enabled:
            self.log_action("Quality gate active; skipping trade execution.", "warning")
            return

        msg = (
            f"Predicted return: {prediction:.6f} (min_edge={self._min_edge():.6f}) -> {decision}"
        )
        self.log_action(msg, "info" if decision != "HOLD" else "warning")
        if self.data.empty:
            self.log_action("No price data available for execution; skipping trade.", "warning")
            return
        price = float(self.data["Close"].iloc[-1])
        trade_summary = self.order_executor.process_signal(asset, decision, price, self.risk_manager)
        if trade_summary.get("status") not in {"noop", "already_long"}:
            self.log_action(f"Paper trade summary: {trade_summary}", "info")
        send_email(
            f"{decision} Signal for {asset} using {strategy_name}",
            msg,
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

        persistence = ModelPersistence()
        artifact = persistence.load(persistence_key)
        if not artifact:
            return (SignalAction.HOLD, None, None, {"reason": "no_model"})

        pipeline = artifact.get("model")
        meta = artifact.get("metadata", {})

        if pipeline is None:
            return (SignalAction.HOLD, None, None, {"reason": "no_pipeline"})

        # Check quality gate
        trade_enabled = self._gate_from_metadata(meta or {})
        if not trade_enabled:
            return (SignalAction.HOLD, None, None, {"reason": "quality_gate_failed"})

        try:
            check_is_fitted(pipeline)
        except NotFittedError:
            return (SignalAction.HOLD, None, None, {"reason": "pipeline_not_fitted"})

        # Build features
        try:
            feature_data, feature_columns = self._build_features(data)
            if len(feature_data) < self.min_inference_rows:
                return (SignalAction.HOLD, None, None, {"reason": "insufficient_data"})

            X_inference = feature_data.iloc[-self.min_inference_rows:][feature_columns]
            prediction = float(pipeline.predict(X_inference)[-1])
        except (ValueError, TypeError, KeyError) as exc:
            return (SignalAction.HOLD, None, None, {"reason": f"inference_failed: {exc}"})

        decision = self._decide_signal(prediction)
        action_map = {"BUY": SignalAction.BUY, "SELL": SignalAction.SELL, "HOLD": SignalAction.HOLD}
        action = action_map.get(decision, SignalAction.HOLD)

        metadata = {
            "strategy_type": "day_trading_ml",
            "predicted_return": prediction,
            "min_edge": self._min_edge(),
            "reason": decision.lower(),
        }

        return (action, None, prediction, metadata)
