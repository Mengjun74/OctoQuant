import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import Strategy
from octoquant.core.features import compute_features


class LightGBMStrategy(Strategy):
    """
    Use a pre-trained LightGBM regression model to predict next-period returns.

    Parameters
    ----------
    model_path : str
        Path to the saved LightGBM model (txt file).
    feature_path : str
        Path to the pickled list of feature column names used in training.
    threshold : float
        Return threshold above which to enter a long position.
    short : bool
        If True, allow taking short positions when prediction < -threshold.
    """

    def __init__(self, model_path: str, feature_path: str, threshold: float = 0.0, short: bool = False):
        self.model = lgb.Booster(model_file=model_path)
        self.feature_names = joblib.load(feature_path)
        self.threshold = threshold
        self.short = short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        features = compute_features(df)
        features = features.reindex(columns=self.feature_names).dropna()

        preds = pd.Series(self.model.predict(features), index=features.index)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[preds > self.threshold] = 1
        if self.short:
            signal.loc[preds < -self.threshold] = -1
        return signal.ffill().fillna(0)


class AutoRegStrategy(Strategy):
    """
    Autoregressive baseline using pre-trained statsmodels AutoReg results.
    """

    def __init__(self, model_path: str, threshold: float = 0.0, short: bool = False):
        self.model = joblib.load(model_path)
        self.threshold = threshold
        self.short = short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        returns = df["Close"].pct_change().dropna()
        if returns.empty:
            return pd.Series(0, index=df.index, dtype=int)

        start = self.model.k_ar
        end = start + len(returns) - 1
        preds = self.model.predict(start=start, end=end, dynamic=True)
        preds = pd.Series(preds, index=returns.index)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[preds.index[preds > self.threshold]] = 1
        if self.short:
            signal.loc[preds.index[preds < -self.threshold]] = -1
        return signal.ffill().fillna(0)
