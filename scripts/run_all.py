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
from octoquant.strat.sma_cross import SmaCross
from octoquant.strat.rsi_pullback import RsiPullback
from octoquant.strat.lwpi_tdfi import LwpiTdfiStrategy
from octoquant.strat.ml_models import LightGBMStrategy, AutoRegStrategy
from octoquant.strat.ensemble import MlTdfiEnsemble
from octoquant.risk.position_sizer import FixedSizer
from octoquant.risk.risk_manager import BasicRisk
from octoquant.backtest.engine import BacktestEngine
from octoquant.backtest.metrics import compute_metrics

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

# Update Strategy wrapper to handle dynamic thresholds if needed
# We can just pre-calc threshold based on distribution or use a percentile strategy
class DynamicLightGBM(LightGBMStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        features = compute_features(df)
        features = features.reindex(columns=self.feature_names).dropna()
        if features.empty: return pd.Series(0, index=df.index)

        preds = pd.Series(self.model.predict(features), index=features.index)
        
        # Dynamic Threshold: Top 10% / Bottom 10%
        # Windowed or Full? Full for backtest simplification
        p95 = preds.quantile(0.95)
        p05 = preds.quantile(0.05)
        
        signal = pd.Series(0, index=df.index, dtype=int)
        signal.loc[preds > p95] = 1
        if self.short:
            signal.loc[preds < p05] = -1
            
        print(f"Dynamic Thresholds: > {p95:.6f} / < {p05:.6f}")
        return signal.ffill().fillna(0)

def run_backtests():
    print("\n--- Running Batch Backtests (Last 6 Months) ---")
    test_start = "2025-07-20"
    test_end = "2026-01-21"
    
    df = fetch_data(test_start, test_end)
    
    strategies = {
        "SMA_Cross": SmaCross(fast=20, slow=60),
        "RSI_Pullback": RsiPullback(),
        "LWPI_TDFI_Optimized": LwpiTdfiStrategy(lwpi_period=135, tdfi_lookback=55, sl_type="atr", atr_mult=2.0),
        "ML_LightGBM_Dynamic": DynamicLightGBM(
            # Using the NEW trained model
            os.path.join(MODELS_DIR, "lightgbm_model.txt"),
            os.path.join(MODELS_DIR, "lightgbm_features.pkl"),
            threshold=0.0, # Ignored by override
            short=True
        ),
        "Ensemble_ML_TDFI": MlTdfiEnsemble(
            model_path=os.path.join(MODELS_DIR, "lightgbm_model.txt"),
            feature_path=os.path.join(MODELS_DIR, "lightgbm_features.pkl"),
            tdfi_lookback=55, # Conservative trend filter
            short=True
        )
    }
    
    ps = FixedSizer(size_pct=0.95)
    risk = BasicRisk(1.0)
    
    results = []
    
    class MockFeed:
        def load(self): return df.copy()

    for name, strat in strategies.items():
        print(f"Testing {name}...")
        try:
            eng = BacktestEngine(MockFeed(), strat, ps, risk, commission_bps=8, slippage_bps=3, initial_cash=1000)
            equity, trades = eng.run()
            
            # Save Output
            csv_path = os.path.join(OUTPUT_DIR, f"{name}_results.csv")
            full_data = pd.DataFrame({"Equity": equity})
            full_data = full_data.join(trades, lsuffix="_t")
            full_data.to_csv(csv_path)
            
            m = compute_metrics(equity, periods_per_year=17520)
            res = {
                "Strategy": name,
                "Return": m["Total Return"],
                "Sharpe": m["Sharpe"],
                "DD": m["Max Drawdown"],
                "Trades": len(trades[trades["turnover"] > 0])
            }
            results.append(res)
        except Exception as e:
            print(f"Failed {name}: {e}")
            import traceback
            traceback.print_exc()
            
    res_df = pd.DataFrame(results).sort_values("Return", ascending=False)
    print("\nFinal Report:")
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(OUTPUT_DIR, "final_summary_ML.csv"), index=False)

if __name__ == "__main__":
    # train_models() # already trained via train_ml.py
    run_backtests()
