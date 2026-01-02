# strategies/strategy_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trading_bot.utils.logger import setup_logger
from trading_bot.utils.paper_trading import PaperTradingExecutor
from trading_bot.utils.risk_management import RiskManager
from trading_bot.utils.seeding import set_global_seeds

if TYPE_CHECKING:
    import logging

    from trading_bot.models.config import StrategyConfig


class StrategyBase(ABC):
    """Common template for all strategies (data prep → validate → indicators → strategy run)."""

    MIN_ROWS: int = 50
    logger: logging.Logger
    risk_manager: RiskManager
    order_executor: PaperTradingExecutor
    seed: int

    def __init__(self, config: StrategyConfig | dict[str, Any], data: pd.DataFrame) -> None:
        self.config = config
        self.data = data
        self.logger = setup_logger(self.__class__.__name__, config.get("log_level") if config else None)
        risk_cfg = None
        if config:
            risk_cfg = (
                getattr(config, "risk_management", None)
                if hasattr(config, "risk_management")
                else config.get("risk_management")
            )
        self.risk_manager = RiskManager(risk_cfg)

        # Use strategy-specific state file to prevent position leakage between strategies
        strategy_name = config.get("strategy") if config else None
        self.order_executor = PaperTradingExecutor(strategy_name=strategy_name)

        self.seed = set_global_seeds(config.get("seed") if config else None)

    def execute(self) -> Any:
        """Template method orchestrating the strategy flow."""
        prepared = self._prepare_data(self.data)
        if not self._validate_data(prepared):
            return None
        enriched = self._add_indicators(prepared)
        return self._run_strategy(enriched)

    def _prepare_data(self, data: pd.DataFrame | None) -> pd.DataFrame | None:
        """Make a sorted copy of input data."""
        if data is None:
            return None
        return data.copy().sort_index()

    def _validate_data(self, data: pd.DataFrame | None) -> bool:
        if data is None or data.empty:
            self.log_action("Input data frame is empty; skipping execution.", "warning")
            return False
        if len(data) < self.MIN_ROWS:
            self.log_action(f"Not enough data rows; need at least {self.MIN_ROWS}, got {len(data)}.", "warning")
            return False
        return True

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Hook for subclasses to add indicators. Default: no-op."""
        return data

    @abstractmethod
    def _run_strategy(self, data: pd.DataFrame) -> Any:
        """Core strategy implementation (training/inference/decisions)."""
        raise NotImplementedError

    def log_action(self, message: str, level: str = "info") -> None:
        getattr(self.logger, level.lower())(message)

    def _log_feature_importance(
        self,
        pipeline: Any,
        feature_columns: list[str],
        top_n: int = 10,
    ) -> list[tuple[str, float]] | None:
        """Log top N most important features for model interpretability.

        Works with sklearn Pipeline containing a model with feature_importances_
        (e.g., XGBoost, RandomForest, GradientBoosting).

        Args:
            pipeline: Fitted sklearn Pipeline with a model step
            feature_columns: List of feature names matching model input
            top_n: Number of top features to log (default: 10)

        Returns:
            List of (feature_name, importance) tuples, or None if not available
        """
        try:
            # Get model from pipeline
            model = None
            if hasattr(pipeline, "named_steps"):
                model = pipeline.named_steps.get("model")
            elif hasattr(pipeline, "steps"):
                # Last step is usually the model
                model = pipeline.steps[-1][1]

            if model is None:
                return None

            # Check if model has feature importances
            if not hasattr(model, "feature_importances_"):
                self.logger.debug("Model does not have feature_importances_ attribute")
                return None

            importances = model.feature_importances_

            # Validate feature columns match importances length
            if len(feature_columns) != len(importances):
                self.logger.warning(
                    f"Feature columns ({len(feature_columns)}) don't match "
                    f"importances ({len(importances)}). Skipping feature importance logging."
                )
                return None

            # Sort by importance (descending)
            sorted_idx = np.argsort(importances)[::-1][:top_n]
            top_features = [(feature_columns[i], float(importances[i])) for i in sorted_idx]

            # Log with formatting
            self.logger.info(f"Top {top_n} feature importances:")
            for rank, (name, imp) in enumerate(top_features, 1):
                self.logger.info(f"  {rank}. {name}: {imp:.4f}")

            return top_features

        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.logger.debug(f"Could not extract feature importance: {exc}")
            return None

    def _calculate_adaptive_threshold(
        self,
        data: pd.DataFrame,
        base_threshold: float,
        volatility_window: int = 20,
        reference_volatility: float | None = 0.02,
        min_multiplier: float = 0.5,
        max_multiplier: float = 3.0,
    ) -> float:
        """Calculate volatility-adjusted decision threshold.

        Adjusts the base threshold based on recent market volatility.
        In high-volatility regimes, thresholds are increased to reduce noise.
        In low-volatility regimes, thresholds are decreased to capture smaller moves.

        Args:
            data: DataFrame with 'Close' column for volatility calculation
            base_threshold: Base threshold from config (e.g., 0.005 for 0.5%)
            volatility_window: Rolling window for volatility calculation (default: 20 days)
            reference_volatility: "Normal" volatility level for scaling (default: 2% daily)
            min_multiplier: Minimum threshold multiplier (default: 0.5)
            max_multiplier: Maximum threshold multiplier (default: 3.0)

        Returns:
            Volatility-adjusted threshold
        """
        if "Close" not in data.columns or len(data) < volatility_window:
            self.logger.debug(
                f"Cannot calculate adaptive threshold: insufficient data "
                f"(need {volatility_window} rows, got {len(data)})"
            )
            return base_threshold

        try:
            # Calculate recent volatility as rolling std of daily returns
            returns = data["Close"].pct_change().dropna()
            if len(returns) < volatility_window:
                return base_threshold

            recent_volatility = float(returns.iloc[-volatility_window:].std())

            if recent_volatility <= 0 or np.isnan(recent_volatility):
                return base_threshold

            if reference_volatility is None or reference_volatility <= 0 or np.isnan(reference_volatility):
                historical_vol = returns.rolling(volatility_window).std().dropna()
                if historical_vol.empty:
                    return base_threshold
                reference_volatility = float(historical_vol.median())
                if reference_volatility <= 0 or np.isnan(reference_volatility):
                    return base_threshold

            # Calculate volatility multiplier (clamped between min and max)
            volatility_multiplier = recent_volatility / reference_volatility
            volatility_multiplier = max(min_multiplier, min(max_multiplier, volatility_multiplier))

            adaptive_threshold = base_threshold * volatility_multiplier

            self.logger.info(
                f"Adaptive threshold: {adaptive_threshold:.4f} "
                f"(base={base_threshold:.4f}, volatility={recent_volatility:.4f}, "
                f"ref_vol={reference_volatility:.4f}, "
                f"multiplier={volatility_multiplier:.2f})"
            )

            return adaptive_threshold

        except (ValueError, TypeError, KeyError) as exc:
            self.logger.debug(f"Error calculating adaptive threshold: {exc}")
            return base_threshold

    def _get_adaptive_thresholds(
        self,
        data: pd.DataFrame,
        use_adaptive: bool | None = None,
    ) -> tuple[float, float]:
        """Get buy and sell thresholds, optionally adjusted for volatility.

        Args:
            data: DataFrame with 'Close' column
            use_adaptive: Override config setting for adaptive thresholds

        Returns:
            Tuple of (buy_threshold, sell_threshold)
        """
        # Get base thresholds from config
        buy_threshold = self.config.get("buy_threshold", 0.005)
        sell_threshold = self.config.get("sell_threshold", -0.005)

        # Check if adaptive thresholds are enabled
        if use_adaptive is None:
            use_adaptive = self.config.get("use_adaptive_thresholds", False)

        if not use_adaptive:
            return buy_threshold, sell_threshold

        # Get adaptive threshold config
        adaptive_config = self.config.get("adaptive_threshold_config", {})
        volatility_window = adaptive_config.get("volatility_window", 20)
        reference_volatility = adaptive_config.get("reference_volatility", 0.02)
        min_multiplier = adaptive_config.get("min_multiplier", 0.5)
        max_multiplier = adaptive_config.get("max_multiplier", 3.0)

        # Calculate adaptive buy threshold
        adaptive_buy = self._calculate_adaptive_threshold(
            data,
            abs(buy_threshold),
            volatility_window=volatility_window,
            reference_volatility=reference_volatility,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )

        # Calculate adaptive sell threshold (keep sign)
        adaptive_sell = -self._calculate_adaptive_threshold(
            data,
            abs(sell_threshold),
            volatility_window=volatility_window,
            reference_volatility=reference_volatility,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )

        return adaptive_buy, adaptive_sell
