import pandas as pd
from typing import Optional

class MACD:
    def __init__(self, data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        """
        MACD indicator.
        :param data: DataFrame containing 'Close'.
        """
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def calculate(self) -> pd.DataFrame:
        """
        Calculate the MACD line, signal line, and histogram without mutating original data.
        """
        df = self.data.copy()
        df['EMA_short'] = df['Close'].ewm(span=self.short_window, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=self.long_window, adjust=False).mean()
        df['MACD'] = df['EMA_short'] - df['EMA_long']
        df['Signal'] = df['MACD'].ewm(span=self.signal_window, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        return df[['MACD', 'Signal', 'MACD_Histogram']]
