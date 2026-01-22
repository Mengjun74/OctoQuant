import argparse
import os
import sys
import yaml


from octoquant.data.feed_ccxt import CCXTDataFeed
from octoquant.strat.sma_cross import SmaCross
from octoquant.strat.rsi_pullback import RsiPullback
from octoquant.strat.lwpi_tdfi import LwpiTdfiStrategy
from octoquant.strat.ml_models import LightGBMStrategy, AutoRegStrategy
from octoquant.risk.position_sizer import VolTargetSizer, FixedSizer
from octoquant.risk.risk_manager import BasicRisk
from octoquant.backtest.engine import BacktestEngine
from octoquant.backtest.metrics import compute_metrics, pretty_print


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_feed(cfg):
    market = cfg.get("market", "crypto")
    symbol = cfg["symbols"][0]
    interval = cfg.get("interval", "1d")
    
    # Check for authentication keys
    api_key_path = "config/coinbase_dp_api_key.json"
    api_key, private_key = None, None
    if os.path.exists(api_key_path):
        try:
            import json
            with open(api_key_path, "r") as f:
                creds = json.load(f)
                api_key = creds.get("name") or creds.get("id") # CDP keys often use 'name' or 'id' as key
                private_key = creds.get("privateKey")
        except Exception as e:
            print(f"Warning: Failed to load API key: {e}")

    # For Sandbox, user requested it. We can add a flag in settings or just force it if detected
    sandbox = cfg.get("sandbox", True) # Default to True as per user request "make coinbase sandbox"

    if market == "stock":
        # Removed YF support for now as per "delete stock stuff" request, 
        # but keeping branch if needed or raising error
        raise ValueError("Stock market support removed. Please use 'crypto'.")
    else:
        # Default to coinbase for this task
        exchange = os.getenv("CCXT_EXCHANGE", "coinbase") 
        return CCXTDataFeed(
            exchange, 
            symbol, 
            timeframe=interval, 
            start_date=cfg.get("start"), 
            end_date=cfg.get("end"),
            api_key=api_key,
            secret=private_key,
            sandbox=sandbox
        )


def build_strategy(cfg):
    strat_cfg = cfg.get("strategy", {})
    name = strat_cfg.get("name", "sma_cross").lower()

    if name == "sma_cross":
        return SmaCross(strat_cfg.get("fast", 20), strat_cfg.get("slow", 60))

    if name == "rsi_pullback":
        return RsiPullback(
            rsi_period=strat_cfg.get("rsi_period", 2),
            oversold_level=strat_cfg.get("oversold_level", 10.0),
            exit_level=strat_cfg.get("exit_level", 50.0),
            trend_period=strat_cfg.get("trend_period", 50),
            slope_threshold=strat_cfg.get("slope_threshold", 0.0),
        )

    if name == "lwpi_tdfi":
        return LwpiTdfiStrategy(
            lwpi_period=strat_cfg.get("lwpi_period", 135),
            lwpi_smooth=strat_cfg.get("lwpi_smooth", 8),
            tdfi_lookback=strat_cfg.get("tdfi_lookback", 21),
            tdfi_mma=strat_cfg.get("tdfi_mma", 5),
            tdfi_smma=strat_cfg.get("tdfi_smma", 15),
            tdfi_n=strat_cfg.get("tdfi_n", 5),
            tp_pct=cfg.get("risk", {}).get("take_profit_pct", 0.03),
            sl_pct=cfg.get("risk", {}).get("stop_loss_pct", 0.03),
        )

    if name == "ml_lightgbm":
        return LightGBMStrategy(
            strat_cfg["model_path"],
            strat_cfg["feature_path"],
            threshold=strat_cfg.get("threshold", 0.0),
            short=strat_cfg.get("short", False),
        )

    if name == "ml_autoreg":
        return AutoRegStrategy(
            strat_cfg["model_path"],
            threshold=strat_cfg.get("threshold", 0.0),
            short=strat_cfg.get("short", False),
        )

    raise ValueError(f"Unsupported strategy: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["backtest", "paper"], help="paper reserved for future use")
    parser.add_argument("--cfg", default="config/settings.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    feed = build_feed(cfg)

    # 策略 / 仓位 / 风控
    # 策略 / 仓位 / 风控
    strat = build_strategy(cfg)
    
    pos_mode = cfg.get("position", {}).get("mode", "vol_target")
    if pos_mode == "fixed":
        ps = FixedSizer(size_pct=cfg.get("position", {}).get("size_pct", 0.95))
    else:
        ps = VolTargetSizer(cfg["position"]["vol_lookback"], cfg["position"]["vol_target_annual"])
    risk = BasicRisk(cfg["max_gross_leverage"])

    # 执行配置（新增）
    execution_cfg = cfg.get("execution", {
        "exec_mode": "close_to_next_open",
        "max_turnover_per_bar": 1.0,
        "min_notional": 0.0,
        "lot_rounding": "none",
    })

    eng = BacktestEngine(
        feed,
        strat,
        ps,
        risk,
        commission_bps=cfg["commission_bps"],
        slippage_bps=cfg["slippage_bps"],
        initial_cash=cfg["backtest"]["initial_cash"],
        execution_cfg=execution_cfg,
    )

    equity, trades = eng.run()

    # 打印尾部净值用于快速 sanity check
    print("\nEquity tail:")
    print(equity.tail())

    # 指标输出
    periods_per_year = cfg.get("metrics", {}).get("periods_per_year", 365)
    rf = cfg.get("metrics", {}).get("rf", 0.0)
    m = compute_metrics(equity, periods_per_year=periods_per_year, rf=rf, returns=trades["ret_net"])
    print("\nMetrics:")
    pretty_print(m)

    return 0


if __name__ == "__main__":
    sys.exit(main())
