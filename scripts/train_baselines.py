import argparse
import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.ar_model import AutoReg

from octoquant.core.features import build_ml_dataset
from octoquant.data.feed_yf import YFDataFeed
from octoquant.data.feed_ccxt import CCXTDataFeed


def build_feed(cfg: dict):
    market = cfg.get("market", "crypto")
    symbol = cfg["symbols"][0]
    interval = cfg.get("interval", "1d")
    if market == "stock":
        return YFDataFeed(symbol, cfg["start"], cfg["end"], interval=interval)
    exchange = os.getenv("CCXT_EXCHANGE", "binance")
    return CCXTDataFeed(exchange, symbol, timeframe=interval)


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def time_split(df: pd.DataFrame, train_ratio: float = 0.7) -> int:
    split_idx = int(len(df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Not enough data for the requested train ratio.")
    return split_idx


def train_lightgbm(dataset: pd.DataFrame, output_dir: Path) -> dict:
    features = dataset.drop(columns=["target"])
    target = dataset["target"]
    split = time_split(dataset)

    X_train, X_valid = features.iloc[:split], features.iloc[split:]
    y_train, y_valid = target.iloc[:split], target.iloc[split:]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    params = {
        "objective": "regression",
        "metric": ["l2", "l1"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(30),
        lgb.log_evaluation(50),
    ]

    booster = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=500,
        callbacks=callbacks,
    )

    preds = booster.predict(X_valid)
    mse = float(((preds - y_valid) ** 2).mean())
    mae = float(np.abs(preds - y_valid).mean())
    ic = float(pd.Series(preds, index=y_valid.index).corr(y_valid))

    model_path = output_dir / "lightgbm_baseline.txt"
    booster.save_model(str(model_path))

    importance = booster.feature_importance(importance_type="gain")
    feature_importance = sorted(
        zip(features.columns, importance),
        key=lambda x: x[1],
        reverse=True,
    )

    joblib.dump(features.columns.tolist(), output_dir / "lightgbm_features.pkl")

    return {
        "model_path": str(model_path),
        "mse": mse,
        "mae": mae,
        "information_coefficient": ic,
        "features": features.columns.tolist(),
        "feature_importance": feature_importance[:10],
        "split_index": split,
    }


def train_autoreg(dataset: pd.DataFrame, output_dir: Path, lags: int = 5) -> dict:
    target = dataset["target"]
    split = time_split(dataset)

    train_series = target.iloc[:split]
    valid_series = target.iloc[split:]

    model = AutoReg(train_series, lags=lags, old_names=False).fit()
    preds = model.predict(
        start=len(train_series),
        end=len(train_series) + len(valid_series) - 1,
        dynamic=False,
    )

    preds = pd.Series(preds, index=valid_series.index)
    mse = float(((preds - valid_series) ** 2).mean())
    mae = float(np.abs(preds - valid_series).mean())
    ic = float(preds.corr(valid_series))

    model_path = output_dir / "autoreg_baseline.pkl"
    with open(model_path, "wb") as f:
        joblib.dump(model, f)

    metadata = {
        "lags": lags,
        "split_index": split,
        "model_path": str(model_path),
        "mse": mse,
        "mae": mae,
        "information_coefficient": ic,
    }
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train baseline ML models.")
    parser.add_argument("--cfg", default="config/settings.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default="models", help="Directory to save models/metadata.")
    parser.add_argument("--lags", type=int, default=5, help="Number of lags for AutoReg baseline.")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    feed = build_feed(cfg)
    df = feed.load()

    dataset = build_ml_dataset(df)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["lightgbm"] = train_lightgbm(dataset, output_dir)
    results["autoreg"] = train_autoreg(dataset, output_dir, lags=args.lags)

    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Baseline training complete.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
