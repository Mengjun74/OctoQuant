import argparse
import itertools
import pandas as pd
import yaml
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from octoquant.data.feed_ccxt import CCXTDataFeed
from octoquant.strat.lwpi_tdfi import LwpiTdfiStrategy
from octoquant.risk.position_sizer import FixedSizer
from octoquant.risk.risk_manager import BasicRisk
from octoquant.backtest.engine import BacktestEngine
from octoquant.backtest.metrics import compute_metrics

def run_grid_search(cfg_path):
    with open(cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    # 1. Load Data ONCE (Speed up)
    print("Loading data...")
    # Using existing config settings for data
    sandbox = base_cfg.get("sandbox", True)
    feed = CCXTDataFeed(
        "coinbase", # Hardcoded for now based on context
        base_cfg["symbols"][0],
        timeframe=base_cfg.get("interval", "30m"),
        start_date=base_cfg.get("start"),
        end_date=base_cfg.get("end"),
        sandbox=sandbox
    )
    df = feed.load()
    print(f"Data Loaded: {len(df)} bars")

    # 2. Define Parameter Grid
    lwpi_periods = [100, 135, 150]
    tdfi_lookbacks = [13, 21, 34, 55]
    
    # 3. Strategy / Engine Setup (Reusable parts)
    ps = FixedSizer(size_pct=0.95)
    risk = BasicRisk(1.0)
    exec_cfg = base_cfg.get("execution", {})

    results = []

    combinations = list(itertools.product(lwpi_periods, tdfi_lookbacks))
    print(f"Starting Grid Search with {len(combinations)} combinations...")

    for lwpi_p, tdfi_l in combinations:
        print(f"Testing LWPI={lwpi_p}, TDFI={tdfi_l}...", end="\r")
        
        # Build Strategy with params
        strat = LwpiTdfiStrategy(
            lwpi_period=lwpi_p,
            tdfi_lookback=tdfi_l,
            tp_pct=0.03,
            sl_pct=0.03,
            sl_type="atr",  # Testing ATR stops as requested
            atr_period=14,
            atr_mult=2.0
        )
        
        # We need to manually inject the PRE-LOADED data into engine 
        # normally engine calls feed.load(). We can subclass or just mock.
        # simpler: modify config or just re-instantiate engine with a MockFeed
        
        class MockFeed:
            def load(self): return df.copy()
            
        eng = BacktestEngine(
            MockFeed(),
            strat,
            ps,
            risk,
            commission_bps=base_cfg["commission_bps"],
            slippage_bps=base_cfg["slippage_bps"],
            initial_cash=base_cfg["backtest"]["initial_cash"],
            execution_cfg=exec_cfg
        )
        
        equity, trades = eng.run()
        
        # Metrics
        m = compute_metrics(equity, periods_per_year=17520, rf=0.0)
        
        res = {
            "LWPI": lwpi_p,
            "TDFI": tdfi_l,
            "Return": m["Total Return"],
            "Sharpe": m["Sharpe"],
            "DD": m["Max Drawdown"],
            "Trades": len(trades[trades["turnover"] > 0])
        }
        results.append(res)

    print("\n\nOptimization Results (Sorted by Return):")
    res_df = pd.DataFrame(results).sort_values("Return", ascending=False)
    print(res_df.to_string(index=False))
    
    # Save to file
    res_df.to_csv("optimization_results.csv", index=False)

if __name__ == "__main__":
    run_grid_search("config/settings.yaml")
