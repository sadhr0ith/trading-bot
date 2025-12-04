# indicators/sma.py
import pandas as pd
from typing import Optional

from indicators.indicator_base import IndicatorBase


class SMA(IndicatorBase):
    def __init__(self, data: pd.DataFrame, window: int = 20, price_col: str = "Close", alias: Optional[str] = None):
        super().__init__(data)
        self.window = window
        self.price_col = price_col
        self.alias = alias

    def calculate(self) -> pd.DataFrame:
        """Calculate SMA without mutating input data."""
        column_name = self.alias or (f"SMA_{self.window}" if self.window else "SMA")
        result = pd.DataFrame(index=self.data.index)
        result[column_name] = self.data[self.price_col].rolling(window=self.window, min_periods=self.window).mean()
        return result
