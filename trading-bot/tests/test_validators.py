import pandas as pd

from trading_bot.utils.validators import ConfigValidator, DataValidator


def test_config_validator_rejects_unknown_indicator():
    config = {
        "strategy": "short_term",
        "data_source": "yahoo",
        "ticker": "AAPL",
        "period": "1y",
        "interval": "1d",
        "indicators": ["unknown"],
    }
    result = ConfigValidator().validate(config)
    assert not result.is_valid
    assert any("Indicators not implemented" in err for err in result.errors)


def test_data_validator_flags_missing_columns():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    result = DataValidator().validate(df)
    assert not result.is_valid
    assert any("Missing required columns" in err for err in result.errors)


def test_data_validator_missing_columns_returns_none_data():
    """Validator should fail fast and not return a mutated dataframe when columns are missing."""
    df = pd.DataFrame({"Price": [1, 2, 3]})
    result = DataValidator(require_ohlcv=True).validate(df)
    assert result.is_valid is False
    assert result.data is None


def test_data_validator_can_keep_nonpositive_volume():
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.1, 2.1],
            "Low": [0.9, 1.9],
            "Close": [1.0, 2.0],
            "Volume": [0.0, -1.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    result = DataValidator(min_rows=1, drop_nonpositive_volume=False).validate(df)
    assert result.is_valid is True
    assert result.data is not None
    assert len(result.data) == 2
