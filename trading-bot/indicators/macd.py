import pandas as pd

class MACD:
    def __init__(self, data, short_window=12, long_window=26, signal_window=9):
        """
        Initialize MACD with the given data.
        :param data: DataFrame containing the price data (assumes 'Close' column exists).
        :param short_window: Period for the short-term EMA (default 12).
        :param long_window: Period for the long-term EMA (default 26).
        :param signal_window: Period for the signal line EMA (default 9).
        """
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def calculate(self):
        """
        Calculate the MACD line, signal line, and histogram.
        :return: DataFrame with MACD, Signal, and MACD_Histogram columns.
        """
        # Calculate short-term and long-term EMAs
        self.data['EMA_short'] = self.data['Close'].ewm(span=self.short_window, adjust=False).mean()
        self.data['EMA_long'] = self.data['Close'].ewm(span=self.long_window, adjust=False).mean()

        # Calculate MACD line
        self.data['MACD'] = self.data['EMA_short'] - self.data['EMA_long']

        # Calculate Signal line
        self.data['Signal'] = self.data['MACD'].ewm(span=self.signal_window, adjust=False).mean()

        # Calculate MACD Histogram
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal']

        return self.data[['MACD', 'Signal', 'MACD_Histogram']]
