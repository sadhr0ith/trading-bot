import numpy as np
import pandas as pd

from trading_bot.strategies.day_trading_ml_strategy import DayTradingMLStrategy


def test_day_trading_ml_target_alignment():
    index = pd.date_range("2024-01-01", periods=60, freq="1H", tz="UTC")
    close = np.linspace(100, 160, num=60)
    data = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": [100.0] * 60,
        },
        index=index,
    )
    config = {
        "strategy": "day_trading_ml",
        "ticker": "TEST",
        "return_horizon": 1,
    }
    strategy = DayTradingMLStrategy(config, data)
    features, _ = strategy._build_features(data)

    expected = data["Close"].pct_change(1).shift(-1).loc[features.index]
    assert "target" in features.columns
    assert np.allclose(features["target"].values, expected.values, equal_nan=False)
