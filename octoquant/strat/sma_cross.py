import pandas as pd
from .base import Strategy

class SmaCross(Strategy):
    def __init__(self, fast=20, slow=60):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"].astype(float)
        f = close.rolling(self.fast).mean()
        s = close.rolling(self.slow).mean()
        raw = (f > s).astype(int) - (f < s).astype(int)
        signal = raw.where(raw.ne(raw.shift())).ffill().fillna(0)
        return signal.astype(int)
