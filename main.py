import argparse
import os
import sys
import yaml

from octoquant.data.feed_yf import YFDataFeed
from octoquant.data.feed_ccxt import CCXTDataFeed
from octoquant.strat.sma_cross import SmaCross
from octoquant.strat.rsi_pullback import RsiPullback
from octoquant.risk.position_sizer import VolTargetSizer
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
    if market == "stock":
        return YFDataFeed(symbol, cfg["start"], cfg["end"], interval=interval)
    else:
        exchange = os.getenv("CCXT_EXCHANGE", "binance")
        return CCXTDataFeed(exchange, symbol, timeframe=interval)


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

    raise ValueError(f"Unsupported strategy: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["backtest", "paper"], help="paper reserved for future use")
    parser.add_argument("--cfg", default="config/settings.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    feed = build_feed(cfg)

    # 策略 / 仓位 / 风控
    strat = build_strategy(cfg)
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
