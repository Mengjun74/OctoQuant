import numpy as np
import pandas as pd

from .base import Strategy


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Classic Wilder RSI implementation."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    # Avoid division by zero; when avg_loss == 0 the RSI should read 100.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.fillna(100.0)  # when avg_loss is zero, treat as max RSI
    return rsi


class RsiPullback(Strategy):
    """
    Long-only RSI(2) pullback strategy with trend filter.

    - Trend filter:  Close above its rolling SMA and SMA slope above threshold.
    - Entry:        RSI <= oversold_level while trend filter holds.
    - Exit:         RSI >= exit_level or trend filter breaks.
    """

    def __init__(
        self,
        rsi_period: int = 2,
        oversold_level: float = 10.0,
        exit_level: float = 50.0,
        trend_period: int = 50,
        slope_threshold: float = 0.0,
    ):
        self.rsi_period = rsi_period
        self.oversold_level = oversold_level
        self.exit_level = exit_level
        self.trend_period = trend_period
        self.slope_threshold = slope_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"].astype(float)

        rsi = _rsi(close, self.rsi_period)
        sma = close.rolling(self.trend_period).mean()
        sma_slope = sma - sma.shift(1)
        trend_ok = (close > sma) & (sma_slope >= self.slope_threshold)

        entries = trend_ok & (rsi <= self.oversold_level)
        exits = (~trend_ok) | (rsi >= self.exit_level)

        actions = pd.Series(np.nan, index=close.index, dtype=float)
        actions.loc[exits] = 0.0
        actions.loc[entries] = 1.0

        signal = actions.ffill().fillna(0.0).astype(int)
        return signal
