import pandas as pd

class RSI:
    def __init__(self, data: pd.DataFrame, period: int = 14):
        """
        Relative Strength Index.
        :param data: DataFrame containing 'Close' column.
        :param period: Lookback window.
        """
        self.data = data
        self.period = period

    def calculate(self) -> pd.Series:
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi
