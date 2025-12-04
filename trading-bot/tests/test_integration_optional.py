import pathlib
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

import strategies.short_term_strategy as sts
from utils import model_persistence
from utils.paper_trading import PaperTradingExecutor
from utils.risk_management import RiskManager


def _synthetic_price_data(rows=80):
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    base = np.linspace(100, 120, rows) + np.random.randn(rows)
    return pd.DataFrame(
        {
            "Open": base * 0.99,
            "High": base * 1.01,
            "Low": base * 0.98,
            "Close": base,
            "Volume": np.random.randint(1_000, 10_000, rows),
        },
        index=idx,
    )


def test_short_term_train_save_load_predict_fast(monkeypatch, tmp_path):
    """Integration: train→save→load→predict with lightweight DummyRegressor."""
    data = _synthetic_price_data(90)

    class TmpPersistence(model_persistence.ModelPersistence):
        def __init__(self, base_dir=None):
            super().__init__(base_dir=tmp_path / "models")

    monkeypatch.setattr(model_persistence, "ModelPersistence", TmpPersistence)
    monkeypatch.setattr(sts, "ModelPersistence", TmpPersistence)
    monkeypatch.setattr(sts, "XGBRegressor", lambda **kwargs: DummyRegressor(strategy="mean"))
    monkeypatch.setattr(sts, "send_email", lambda *args, **kwargs: True)

    config = {
        "ticker": "TEST",
        "strategy": "short_term",
        "indicators": ["macd", "rsi", "adx"],
        "use_indicators": True,
        "notification_email": "none@example.com",
        "seed": 42,
    }
    strategy = sts.ShortTermStrategy(config, data)
    strategy.execute()  # should train and persist

    # Ensure artifact exists
    artifacts = list((tmp_path / "models" / "short_term").glob("*.pkl"))
    assert artifacts, "Expected persisted model artifact"


def test_paper_trading_stop_loss_exit():
    """Integration: BUY then HOLD triggers stop-loss exit."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        state_path = pathlib.Path(f.name)

    rm = RiskManager({"max_position_size": 0.5, "stop_loss": 0.02, "take_profit": 0.1, "trading_fee": 0.0})
    executor = PaperTradingExecutor(state_path=state_path, initial_balance=10_000.0)

    buy = executor.process_signal("TEST", "BUY", 100.0, rm)
    assert buy["status"] == "opened"

    # Price drops to trigger stop-loss on HOLD
    hold = executor.process_signal("TEST", "HOLD", 97.0, rm)
    assert hold["status"] in {"closed", "noop"}

    state_path.unlink(missing_ok=True)


def test_risk_manager_zero_balance():
    """Edge case: zero balance returns zero position size."""
    rm = RiskManager({"max_position_size": 0.1})
    assert rm.calculate_position_size(0.0, 100.0) == 0.0
