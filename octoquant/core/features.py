import numpy as np
import pandas as pd


def compute_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    volume_col: str = "Volume",
    price_windows=(5, 10, 20),
    vol_windows=(5, 10, 20),
) -> pd.DataFrame:
    """
    Build a tabular feature set from OHLCV bars.

    Features include:
    - lagged returns
    - moving-average ratios
    - realized volatility
    - RSI
    - volume z-scores
    """
    close = df[price_col].astype(float)
    volume = df[volume_col].astype(float) if volume_col in df else None

    feats = {}

    ret1 = close.pct_change().fillna(0.0)
    feats["ret_1"] = ret1

    for w in price_windows:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        feats[f"ma_ratio_{w}"] = close / ma - 1.0
        feats[f"vol_{w}"] = std / close
        feats[f"ret_{w}"] = close.pct_change(w)

    if volume is not None:
        vol_ret = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        feats["volume_ret_1"] = vol_ret.fillna(0.0)
        for w in vol_windows:
            vol_ma = volume.rolling(w).mean()
            vol_std = volume.rolling(w).std()
            feats[f"volume_z_{w}"] = (volume - vol_ma) / vol_std

    feats["rsi_14"] = rsi(close, period=14)
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
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.fillna(50.0)  # neutral when insufficient history
    return rsi


def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper that adds the next-period return target.
    """
    features = compute_features(df)
    close = df["Close"].astype(float)
    target = close.pct_change().shift(-1)
    dataset = features.join(target.rename("target")).dropna()
    return dataset
