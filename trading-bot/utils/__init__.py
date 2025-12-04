"""Utilities module.

This module contains utility functions and classes for:
- Data validation (validators.py)
- Model persistence (model_persistence.py)
- Paper trading execution (paper_trading.py)
- Risk management (risk_management.py)
- Feature transformers (transformers.py)
- Feature engineering (feature_engineering.py)
- Email notifications (email_notifications.py)
- Logging setup (logger.py)
- Strategy helpers (strategy_helpers.py)
"""

from utils.validators import ConfigValidator, DataValidator, ValidationResult
from utils.model_persistence import ModelPersistence
from utils.paper_trading import PaperTradingExecutor
from utils.risk_management import RiskManager, Position
from utils.logger import setup_logger
from utils.strategy_helpers import train_or_load_pipeline
from utils.feature_engineering import (
    add_lag_features,
    create_forward_return_target,
    build_lag_feature_columns,
    add_indicator_columns,
)

__all__ = [
    "ConfigValidator",
    "DataValidator",
    "ValidationResult",
    "ModelPersistence",
    "PaperTradingExecutor",
    "RiskManager",
    "Position",
    "setup_logger",
    "train_or_load_pipeline",
    "add_lag_features",
    "create_forward_return_target",
    "build_lag_feature_columns",
    "add_indicator_columns",
]
