import pandas as pd

from utils.validators import ConfigValidator, DataValidator


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
