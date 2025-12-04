import pandas as pd

from indicators.indicator_base import IndicatorBase


class ADX(IndicatorBase):
    """Average Directional Index with +DI and -DI."""

    def __init__(self, data: pd.DataFrame, window: int = 14):
        super().__init__(data)
        self.window = window

    def calculate(self) -> pd.DataFrame:
        df = self.data.copy()
        if df.empty or 'High' not in df or 'Low' not in df or 'Close' not in df:
            return pd.DataFrame(index=df.index)

        high = df['High']
        low = df['Low']
        close = df['Close']

        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        tr1 = (high - low).abs()
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(self.window, min_periods=self.window).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / self.window, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / self.window, adjust=False).mean() / atr)

        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()).fillna(0)
        adx = dx.ewm(alpha=1 / self.window, adjust=False).mean()

        out = pd.DataFrame(index=df.index)
        out['Plus_DI'] = plus_di
        out['Minus_DI'] = minus_di
        out['ADX'] = adx
        return out
