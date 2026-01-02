import pandas as pd

from trading_bot.utils.data_health import data_health_report, feature_shift_report


def test_data_health_reports_gaps():
    index = pd.date_range("2024-01-01", periods=5, freq="1H", tz="UTC")
    index = index.delete(2)
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [1, 2, 3, 4],
            "Low": [1, 2, 3, 4],
            "Close": [1, 2, 3, 4],
            "Volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )

    report = data_health_report(data, "1h")
    assert report.get("status") == "ok"
    assert report.get("gap_count", 0) >= 1


def test_feature_shift_report():
    index = pd.date_range("2024-01-01", periods=500, freq="1H", tz="UTC")
    close = pd.Series(range(500), index=index)
    data = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": [1.0] * 500,
        }
    )

    report = feature_shift_report(data, window=200)
    assert report is not None
    assert "return_mean_shift" in report
    assert "return_vol_shift" in report
