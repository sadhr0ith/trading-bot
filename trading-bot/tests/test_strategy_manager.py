import pandas as pd
import pytest

from trading_bot.strategy_manager import select_strategy


def test_select_strategy_invalid():
    with pytest.raises(ValueError):
        select_strategy({"strategy": "unknown"}, pd.DataFrame())


def test_select_strategy_returns_base():
    df = pd.DataFrame({"Close": [1, 2, 3], "Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3], "Volume": [1, 1, 1]})
    strategy = select_strategy({"strategy": "short_term", "indicators": [], "ticker": "T", "data_source": "yahoo", "period": "1d", "interval": "1d"}, df)
    assert hasattr(strategy, "execute")
