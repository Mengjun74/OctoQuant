import numpy as np
import pandas as pd

class VolTargetSizer:
    def __init__(self, lookback=30, ann_vol=0.15):
        self.lookback = lookback
        self.ann_vol = ann_vol

    def target_weights(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        ret = df["Close"].pct_change()
        vol = ret.rolling(self.lookback).std() * np.sqrt(252)
        w_raw = (self.ann_vol / vol.replace(0, np.nan)) * signal
        return w_raw.clip(-1, 1).fillna(0)
