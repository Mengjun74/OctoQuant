import argparse, yaml, os, sys
from octoquant.data.feed_yf import YFDataFeed
from octoquant.data.feed_ccxt import CCXTDataFeed
from octoquant.strat.sma_cross import SmaCross
from octoquant.risk.position_sizer import VolTargetSizer
from octoquant.risk.risk_manager import BasicRisk
from octoquant.backtest.engine import BacktestEngine
from octoquant.backtest.metrics import compute_metrics, pretty_print

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_feed(cfg):
    market = cfg.get("market","crypto")
    symbol = cfg["symbols"][0]
    interval = cfg.get("interval","1d")
    if market == "stock":
        return YFDataFeed(symbol, cfg["start"], cfg["end"], interval=interval)
    else:
        exchange = os.getenv("CCXT_EXCHANGE","binance")
        return CCXTDataFeed(exchange, symbol, timeframe=interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["backtest","paper"], help="paper reserved for future use")
    parser.add_argument("--cfg", default="config/settings.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    feed = build_feed(cfg)

    strat = SmaCross(cfg["strategy"]["fast"], cfg["strategy"]["slow"])
    ps = VolTargetSizer(cfg["position"]["vol_lookback"], cfg["position"]["vol_target_annual"])
    risk = BasicRisk(cfg["max_gross_leverage"])

    eng = BacktestEngine(feed, strat, ps, risk, cfg["commission_bps"], cfg["slippage_bps"], cfg["backtest"]["initial_cash"])
    equity, trades = eng.run()
    m = compute_metrics(
        equity,
        periods_per_year=cfg["metrics"]["periods_per_year"],
        rf=cfg["metrics"].get("rf", 0.0),
    )
    pretty_print(m)
    print(equity.tail())
    return 0

if __name__ == "__main__":
    sys.exit(main())
