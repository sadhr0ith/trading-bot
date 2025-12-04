import numpy as np
import pandas as pd

from indicators.indicator_base import IndicatorBase


class StochasticOscillator(IndicatorBase):
    def __init__(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        super().__init__(data)
        self.k_period = k_period
        self.d_period = d_period

    def calculate(self) -> pd.DataFrame:
        """Calculate Stochastic Oscillator without mutating input data."""
        lowest_low = self.data['Low'].rolling(window=self.k_period, min_periods=self.k_period).min()
        highest_high = self.data['High'].rolling(window=self.k_period, min_periods=self.k_period).max()

        range_ = highest_high - lowest_low
        range_ = range_.replace(0, np.nan)

        percent_k = 100 * (self.data['Close'] - lowest_low) / range_
        percent_d = percent_k.rolling(window=self.d_period, min_periods=self.d_period).mean()

        result = pd.DataFrame({
            'Stochastic_K': percent_k,
            'Stochastic_D': percent_d
        })
        return result
