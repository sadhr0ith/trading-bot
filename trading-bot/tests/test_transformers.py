import numpy as np
import pandas as pd

from trading_bot.indicators.adx import ADX
from trading_bot.utils.transformers import (
    CalendarFeatureTransformer,
    FeatureSelector,
    IndicatorLagTransformer,
    ReturnOutlierClipper,
)


def test_calendar_feature_transformer_adds_fields():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Close": [1, 2, 3]}, index=idx)
    transformer = CalendarFeatureTransformer()

    out = transformer.transform(df)

    for col in ["day_of_week", "month", "is_month_start", "is_month_end"]:
        assert col in out.columns


def test_indicator_lag_transformer_renames_di_columns():
    df = pd.DataFrame(
        {
            "MACD": [0.1, 0.2, 0.3],
            "DI+": [10, 11, 12],
            "DI-": [8, 7, 6],
        }
    )
    transformer = IndicatorLagTransformer(
        indicator_columns=["MACD", "Plus_DI", "Minus_DI"],
        lags=[1],
        column_mapping={"DI+": "Plus_DI", "DI-": "Minus_DI"},
    )

    out = transformer.transform(df)
    assert "Plus_DI_lag_1" in out.columns
    assert "Minus_DI_lag_1" in out.columns


def test_feature_selector_ffill_does_not_bfill():
    df = pd.DataFrame({"Close": [None, 1.0, None, 3.0]})
    selector = FeatureSelector(feature_columns=["Close"], handle_missing="ffill")

    out = selector.transform(df)

    assert out["Close"].iloc[0] == 0.0
    assert out["Close"].iloc[1] == 1.0
    assert out["Close"].iloc[2] == 1.0


def test_feature_selector_adds_missing_flags():
    df = pd.DataFrame({"Close": [None, 1.0, None]})
    selector = FeatureSelector(feature_columns=["Close"], handle_missing="ffill", add_missing_flags=True)

    out = selector.transform(df)

    assert "Close_missing" in out.columns
    assert out["Close_missing"].tolist() == [1, 0, 1]


def test_return_outlier_clipper_clips_extreme_return():
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 1000.0]})
    clipper = ReturnOutlierClipper(lower_pct=0.1, upper_pct=90.0, min_periods=2)

    out = clipper.transform(df)

    assert np.allclose(out["Close"].iloc[:3], df["Close"].iloc[:3])
    assert out["Close"].iloc[-1] < 200.0


def test_adx_outputs_expected_columns():
    data = pd.DataFrame(
        {
            "High": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            "Low": [9, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 21],
            "Close": [9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 19, 20, 21, 22],
        }
    )
    adx = ADX(data)
    out = adx.calculate()
    for col in ["ADX", "Plus_DI", "Minus_DI"]:
        assert col in out.columns
