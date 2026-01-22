import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from octoquant.data.feed_ccxt import CCXTDataFeed
from octoquant.core.features import compute_features

MODELS_DIR = "d:/projects/OctoQuant/models"

def debug_models():
    print("--- Debugging ML Models ---")
    # Fetch a small chunk of recent data
    start_date = "2025-10-01"
    end_date = "2026-01-01"
    
    feed = CCXTDataFeed(
        "coinbase", 
        "BTC/USDT", 
        timeframe="30m",
        start_date=start_date,
        end_date=end_date,
        sandbox=True
    )
    df = feed.load()
    print(f"Data Loaded: {len(df)} records")
    
    # 1. Feature Check
    features = compute_features(df)
    print("Feature Stats:")
    print(features.describe().transpose()[['mean', 'std', 'min', 'max']])
    
    # 2. LightGBM Check
    print("\n[LightGBM Analysis]")
    try:
        model_path = os.path.join(MODELS_DIR, "lightgbm_model.txt")
        feature_path = os.path.join(MODELS_DIR, "lightgbm_features.pkl")
        
        bst = lgb.Booster(model_file=model_path)
        feat_names = joblib.load(feature_path)
        
        # Check alignment
        X = features.reindex(columns=feat_names).dropna()
        if X.empty:
            print("(!) No valid feature rows after reindexing/dropna.")
        else:
            preds = bst.predict(X)
            p_series = pd.Series(preds)
            print("Prediction Stats:")
            print(p_series.describe())
            print(f"Predictions > 0.0005: {(p_series > 0.0005).sum()}")
            print(f"Predictions < -0.0005: {(p_series < -0.0005).sum()}")
    except Exception as e:
        print(f"LGBM Error: {e}")

    # 3. AutoReg Check
    print("\n[AutoReg Analysis]")
    try:
        model_path = os.path.join(MODELS_DIR, "autoreg_model.pkl")
        ar_model = joblib.load(model_path)
        print(f"Lag Lags: {ar_model.k_ar}")
        
    except Exception as e:
        print(f"AutoReg Error: {e}")

if __name__ == "__main__":
    debug_models()
