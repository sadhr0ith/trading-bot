# indicators/bollinger_bands.py
import numpy as np
import pandas as pd

from indicators.indicator_base import IndicatorBase


class BollingerBands(IndicatorBase):
    def __init__(self, data: pd.DataFrame, window: int = 20, num_std: float = 2.0):
        super().__init__(data)
        self.window = window
        self.num_std = num_std

    def calculate(self) -> pd.DataFrame:
        """Calculate Bollinger Bands without mutating input data."""
        rolling_mean = self.data["Close"].rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = self.data["Close"].rolling(window=self.window, min_periods=self.window).std(ddof=0)

        upper = rolling_mean + self.num_std * rolling_std
        lower = rolling_mean - self.num_std * rolling_std
        width = (upper - lower) / rolling_mean.replace(0, np.nan)

        bands = pd.DataFrame({"BB_Middle": rolling_mean, "BB_Upper": upper, "BB_Lower": lower, "BB_Width": width})
        return bands
