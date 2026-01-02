from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_bot.strategies.long_term_strategy import LongTermStrategy
from trading_bot.strategies.mid_term_strategy import MidTermStrategy
from trading_bot.strategies.short_term_strategy import ShortTermStrategy
from trading_bot.utils.feature_engineering import add_lag_features, build_lag_feature_columns
from trading_bot.utils.logger import setup_logger
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.strategy_helpers import build_persistence_key
from trading_bot.utils.transformers import FeatureSelector


class _DummyPaperExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def process_signal(self, *args, **kwargs):
        return {"status": "noop"}


def _monkeypatch_persistence(monkeypatch, tmp_path):
    def _init(self, base_dir=None):
        self.base_dir = tmp_path / "models"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__)

    monkeypatch.setattr(ModelPersistence, "__init__", _init, raising=True)


def _persist_dummy_pipeline(
    strategy_name: str,
    feature_columns: list[str],
    tmp_path,
    ticker: str = "TEST",
    interval: str = "1d",
    data_source: str = "yahoo",
):
    persistence = ModelPersistence(base_dir=tmp_path / "models")
    persistence_key = build_persistence_key(
        strategy=strategy_name,
        data_source=data_source,
        ticker=ticker,
        interval=interval,
    )
    preprocess = ColumnTransformer([("num", StandardScaler(), feature_columns)], remainder="drop")
    pipe = Pipeline([("preprocess", preprocess), ("model", DummyRegressor(strategy="mean"))])
    # Fit on minimal dummy data
    X_fit = pd.DataFrame({col: [1.0, 2.0, 3.0] for col in feature_columns})
    y_fit = pd.Series([0.1, 0.2, 0.3])
    pipe.fit(X_fit, y_fit)
    persistence.save(persistence_key, pipe, scaler=None, metadata={"feature_columns": feature_columns})


def _persist_dummy_short_term_pipeline(
    tmp_path,
    ticker: str = "TEST",
    interval: str = "1d",
    data_source: str = "yahoo",
):
    persistence = ModelPersistence(base_dir=tmp_path / "models")
    persistence_key = build_persistence_key(
        strategy="short_term",
        data_source=data_source,
        ticker=ticker,
        interval=interval,
    )
    pipe = Pipeline(
        [
            ("selector", FeatureSelector(feature_columns=["Close"], handle_missing="ffill")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    X_fit = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    y_fit = pd.Series([0.1, 0.2, 0.3])
    pipe.fit(X_fit, y_fit)
    persistence.save(persistence_key, pipe, scaler=None, metadata={"feature_columns": ["Close"]})


def _persist_constant_pipeline(
    strategy_name: str,
    feature_columns: list[str],
    tmp_path,
    constant: float,
    ticker: str = "TEST",
    interval: str = "1d",
    data_source: str = "yahoo",
):
    persistence = ModelPersistence(base_dir=tmp_path / "models")
    persistence_key = build_persistence_key(
        strategy=strategy_name,
        data_source=data_source,
        ticker=ticker,
        interval=interval,
    )
    preprocess = ColumnTransformer([("num", StandardScaler(), feature_columns)], remainder="drop")
    pipe = Pipeline([("preprocess", preprocess), ("model", DummyRegressor(strategy="constant", constant=constant))])
    X_fit = pd.DataFrame({col: [1.0, 2.0, 3.0] for col in feature_columns})
    y_fit = pd.Series([constant] * len(X_fit))
    pipe.fit(X_fit, y_fit)
    persistence.save(persistence_key, pipe, scaler=None, metadata={"feature_columns": feature_columns})


@pytest.fixture(autouse=True)
def _patch_paper_executor(monkeypatch):
    from trading_bot.strategies import strategy_base
    from trading_bot.utils import paper_trading

    monkeypatch.setattr(paper_trading, "PaperTradingExecutor", _DummyPaperExecutor, raising=True)
    monkeypatch.setattr(strategy_base, "PaperTradingExecutor", _DummyPaperExecutor, raising=True)


def test_mid_term_inference_only(monkeypatch, tmp_path):
    _monkeypatch_persistence(monkeypatch, tmp_path)
    # Create dummy data
    dates = pd.date_range(end=datetime.now(), periods=200, freq="D")
    base = pd.DataFrame(
        {
            "Open": range(1, 201),
            "High": range(2, 202),
            "Low": range(0, 200),
            "Close": range(1, 201),
            "Volume": [1000] * 200,
        },
        index=dates,
    )
    lags = [5, 10, 20, 60, 120]
    data = add_lag_features(base.copy(), lags)
    feature_columns = build_lag_feature_columns(lags)
    _persist_dummy_pipeline("mid_term", feature_columns, tmp_path, interval="1d")

    config = {
        "ticker": "TEST",
        "strategy": "mid_term",
        "data_source": "yahoo",
        "period": "1y",
        "interval": "1d",
        "indicators": [],
        "use_indicators": False,
        "inference_only": True,
    }
    strat = MidTermStrategy(config, data)
    # Should run inference-only path without raising
    strat._run_strategy(data)


def test_long_term_inference_only(monkeypatch, tmp_path):
    _monkeypatch_persistence(monkeypatch, tmp_path)
    # Create dummy data
    dates = pd.date_range(end=datetime.now(), periods=300, freq="D")
    base = pd.DataFrame(
        {
            "Open": range(1, 301),
            "High": range(2, 302),
            "Low": range(0, 300),
            "Close": range(1, 301),
            "Volume": [2000] * 300,
        },
        index=dates,
    )
    lags = [20, 60, 120, 250]
    data = add_lag_features(base.copy(), lags)
    feature_columns = build_lag_feature_columns(lags)
    _persist_dummy_pipeline("long_term", feature_columns, tmp_path, interval="1wk")

    config = {
        "ticker": "TEST",
        "strategy": "long_term",
        "data_source": "yahoo",
        "period": "5y",
        "interval": "1wk",
        "indicators": [],
        "use_indicators": False,
        "inference_only": True,
    }
    strat = LongTermStrategy(config, data)
    # Should run inference-only path without raising
    strat._run_strategy(data)


def test_short_term_inference_only(monkeypatch, tmp_path):
    _monkeypatch_persistence(monkeypatch, tmp_path)
    _persist_dummy_short_term_pipeline(tmp_path, interval="1d")

    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    data = pd.DataFrame(
        {
            "Open": range(1, 31),
            "High": range(2, 32),
            "Low": range(0, 30),
            "Close": range(1, 31),
            "Volume": [500] * 30,
        },
        index=dates,
    )

    config = {
        "ticker": "TEST",
        "strategy": "short_term",
        "data_source": "yahoo",
        "period": "1y",
        "interval": "1d",
        "indicators": [],
        "use_indicators": False,
        "inference_only": True,
        "notification_email": None,
    }

    strat = ShortTermStrategy(config, data)
    strat._run_strategy(data)


def test_short_term_inference_only_min_rows(monkeypatch, tmp_path):
    """Minimal inference-only path uses persisted pipeline without training."""
    _monkeypatch_persistence(monkeypatch, tmp_path)
    _persist_dummy_short_term_pipeline(tmp_path, interval="1d")
    monkeypatch.setattr("trading_bot.strategies.short_term_strategy.send_email", lambda *_, **__: True)

    decisions: list[str] = []

    class _CapturePaperExecutor(_DummyPaperExecutor):
        def process_signal(self, asset, decision, price, rm):
            decisions.append(decision)
            return {"status": "noop", "decision": decision, "asset": asset, "price": price}

    # Patch executors to capture decisions
    from trading_bot.strategies import strategy_base
    from trading_bot.utils import paper_trading

    monkeypatch.setattr(paper_trading, "PaperTradingExecutor", _CapturePaperExecutor, raising=True)
    monkeypatch.setattr(strategy_base, "PaperTradingExecutor", _CapturePaperExecutor, raising=True)

    data = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0.5, 1, 2, 3, 4],
            "Close": [1, 2, 3, 4, 5],
            "Volume": [100] * 5,
        },
        index=pd.date_range(end=datetime.now(), periods=5, freq="D"),
    )

    config = {
        "ticker": "TEST",
        "strategy": "short_term",
        "data_source": "yahoo",
        "period": "1y",
        "interval": "1d",
        "indicators": [],
        "use_indicators": False,
        "inference_only": True,
        "min_inference_rows": 5,
        "notification_email": "none@example.com",
    }

    strat = ShortTermStrategy(config, data)
    strat._run_strategy(data)

    assert decisions, "PaperTradingExecutor should be invoked in inference-only path"


def test_long_term_sell_threshold_respected(monkeypatch, tmp_path):
    """Inference-only long-term respects asymetryczny próg SELL."""
    _monkeypatch_persistence(monkeypatch, tmp_path)
    decisions: list[str] = []

    class _CapturePaperExecutor(_DummyPaperExecutor):
        def process_signal(self, asset, decision, price, rm):
            decisions.append(decision)
            return {"status": "noop", "decision": decision, "asset": asset, "price": price}

    monkeypatch.setattr("trading_bot.utils.paper_trading.PaperTradingExecutor", _CapturePaperExecutor, raising=True)
    monkeypatch.setattr("trading_bot.strategies.strategy_base.PaperTradingExecutor", _CapturePaperExecutor, raising=True)
    monkeypatch.setattr("trading_bot.strategies.long_term_strategy.send_email", lambda *_, **__: True)

    dates = pd.date_range(end=datetime.now(), periods=300, freq="D")
    base = pd.DataFrame(
        {
            "Open": range(1, 301),
            "High": range(2, 302),
            "Low": range(0, 300),
            "Close": range(1, 301),
            "Volume": [2000] * 300,
        },
        index=dates,
    )
    lags = [20, 60, 120, 250]
    data = add_lag_features(base.copy(), lags)
    feature_columns = build_lag_feature_columns(lags)
    _persist_constant_pipeline("long_term", feature_columns, tmp_path, constant=-0.03, interval="1wk")

    config = {
        "ticker": "TEST",
        "strategy": "long_term",
        "data_source": "yahoo",
        "period": "5y",
        "interval": "1wk",
        "indicators": [],
        "use_indicators": False,
        "inference_only": True,
        "buy_threshold": 0.02,
        "sell_threshold": -0.02,
        "notification_email": None,
    }

    strat = LongTermStrategy(config, data)
    strat._run_strategy(data)

    assert decisions and decisions[-1] == "SELL"
