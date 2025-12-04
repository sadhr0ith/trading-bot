# indicators/ema.py
import pandas as pd
from typing import Optional

from indicators.indicator_base import IndicatorBase


class EMA(IndicatorBase):
    def __init__(self, data: pd.DataFrame, span: int = 20, price_col: str = "Close", alias: Optional[str] = None):
        super().__init__(data)
        self.span = span
        self.price_col = price_col
        self.alias = alias

    def calculate(self) -> pd.DataFrame:
        """Calculate EMA without mutating input data."""
        column_name = self.alias or (f"EMA_{self.span}" if self.span else "EMA")
        result = pd.DataFrame(index=self.data.index)
        result[column_name] = self.data[self.price_col].ewm(span=self.span, adjust=False).mean()
        return result
