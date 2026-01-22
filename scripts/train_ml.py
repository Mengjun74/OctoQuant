import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
import sys
from statsmodels.tsa.ar_model import AutoReg

# Add project root to sys.path
sys.path.append(os.getcwd())

from octoquant.data.feed_ccxt import CCXTDataFeed
from octoquant.core.features import compute_features

OUTPUT_DIR = "d:/projects/OctoQuant/output_data"
MODELS_DIR = "d:/projects/OctoQuant/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def fetch_data(start_date, end_date):
    print(f"Fetching data from {start_date} to {end_date}...")
    feed = CCXTDataFeed(
        "coinbase", 
        "BTC/USDT", 
        timeframe="30m",
        start_date=start_date,
        end_date=end_date,
        sandbox=True
    )
    return feed.load()

def train_and_evaluate_ml():
    print("--- Training ML Models (Refined) ---")
    
    # 2.5 Years Training
    train_start = "2023-01-20"
    train_end = "2025-07-20"
    
    df = fetch_data(train_start, train_end)
    features = compute_features(df)
    
    # Target: Next Return
    # We will Standardize the target to help Regression? 
    # Or just use raw returns (small numbers). LightGBM handles scale well.
    # But let's verify target exists.
    target = df["Close"].pct_change().shift(-1)
    
    data = features.join(target.rename("target")).dropna()
    print(f"Training Data Samples: {len(data)}")
    
    # Split Train/Val (Last 20% for early stopping)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]
    X_val = val_data.drop(columns=["target"])
    y_val = val_data["target"]
    
    # 1. LightGBM Refined
    print("Training LightGBM...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    gbm.save_model(os.path.join(MODELS_DIR, "lightgbm_model.txt"))
    joblib.dump(X_train.columns.tolist(), os.path.join(MODELS_DIR, "lightgbm_features.pkl"))
    print("LightGBM Saved.")

if __name__ == "__main__":
    train_and_evaluate_ml()
