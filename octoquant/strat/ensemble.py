import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from octoquant.strat.base import Strategy
from octoquant.core.features import compute_features

class MlTdfiEnsemble(Strategy):
    """
    Ensemble Strategy:
    1. Generates signals using LightGBM dynamic percentile thresholds.
    2. Filters signals using TDFI (Trend Direction Force Index) to ensure trend alignment.
    
    Logic:
    - Long if (ML_Score > 95th_Pct) AND (TDFI > 0.05)
    - Short if (ML_Score < 5th_Pct) AND (TDFI < -0.05)
    """

    def __init__(self, model_path: str, feature_path: str, 
                 tdfi_lookback=55, tdfi_mma=5, tdfi_smma=15, 
                 short: bool = True):
        # ML Setup
        self.model = lgb.Booster(model_file=model_path)
        self.feature_names = joblib.load(feature_path)
        self.short = short
        
        # TDFI Setup
        self.tdfi_lookback = tdfi_lookback
        self.tdfi_mma = tdfi_mma
        self.tdfi_smma = tdfi_smma

    def calc_tdfi(self, df: pd.DataFrame) -> pd.Series:
        """Helper to calculate TDFI series only."""
        close = df["Close"]
        change = close - close.shift(1)
        force = change * df["Volume"]
        abs_force = change.abs() * df["Volume"]
        
        s_force = force.ewm(span=self.tdfi_lookback).mean()
        s_abs_force = abs_force.ewm(span=self.tdfi_lookback).mean()
        
        tdfi_raw = s_force / s_abs_force.replace(0, 1)
        tdfi_smooth = tdfi_raw.ewm(span=self.tdfi_mma).mean()
        tdfi_final = tdfi_smooth.ewm(span=self.tdfi_smma).mean()
        return tdfi_final

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # 1. ML Signals
        features = compute_features(df)
        features = features.reindex(columns=self.feature_names).dropna()
        
        # Align indices
        common_idx = features.index.intersection(df.index)
        if common_idx.empty: return pd.Series(0, index=df.index)
        
        preds = pd.Series(self.model.predict(features.loc[common_idx]), index=common_idx)
        
        # Dynamic Thresholds (Distribution-based)
        p95 = preds.quantile(0.95)
        p05 = preds.quantile(0.05)
        
        # 2. TDFI Filter
        tdfi = self.calc_tdfi(df).loc[common_idx]
        
        # 3. Combine
        signal = pd.Series(0, index=df.index, dtype=int)
        
        # Long: ML Bullish + Trend Bullish
        long_mask = (preds > p95) & (tdfi > 0.05)
        signal.loc[long_mask.index[long_mask]] = 1
        
        if self.short:
            # Short: ML Bearish + Trend Bearish
            short_mask = (preds < p05) & (tdfi < -0.05)
            signal.loc[short_mask.index[short_mask]] = -1
            
        return signal.ffill().fillna(0)
