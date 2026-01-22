import numpy as np
import pandas as pd

def compute_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    volume_col: str = "Volume",
    price_windows=(5, 10, 20, 50),
    vol_windows=(5, 10, 20),
) -> pd.DataFrame:
    """
    Build a tabular feature set from OHLCV bars.
    Enhanced with: MACD, ATR, Slope, Time.
    """
    close = df[price_col].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df[volume_col].astype(float) if volume_col in df else None

    feats = {}

    # 1. Basic Lags
    feats["ret_1"] = close.pct_change().fillna(0.0)
    
    # 2. Moving Averages & Volatility
    for w in price_windows:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        feats[f"dist_ma_{w}"] = close / ma - 1.0
        feats[f"vol_{w}"] = std / close
        feats[f"ret_{w}"] = close.pct_change(w)
        
        # Slope (normalized)
        feats[f"slope_{w}"] = (close - close.shift(w)) / close.shift(w)

    # 3. Volume
    if volume is not None:
        vol_ret = volume.pct_change().replace([np.inf, -np.inf], 0.0)
        feats["vol_ret_1"] = vol_ret.fillna(0.0)
        for w in vol_windows:
            vol_ma = volume.rolling(w).mean()
            vol_std = volume.rolling(w).std()
            feats[f"vol_z_{w}"] = (volume - vol_ma) / (vol_std + 1e-9)

    # 4. Technical Indicators
    # RSI
    feats["rsi_14"] = rsi(close, period=14) / 100.0 # Normalize 0-1
    
    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd_hist"] = (macd - signal) / close # Normalize by price

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feats["atr_14_pct"] = atr / close

    # 5. Time Features (Cyclical)
    if isinstance(df.index, pd.DatetimeIndex):
        day_of_week = df.index.dayofweek
        hour = df.index.hour
        feats["sin_hour"] = np.sin(2 * np.pi * hour / 24)
        feats["cos_hour"] = np.cos(2 * np.pi * hour / 24)

    feature_df = pd.DataFrame(feats, index=df.index)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feature_df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_val = 100.0 - 100.0 / (1.0 + rs)
    return rsi_val.fillna(50.0)
