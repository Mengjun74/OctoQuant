import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, data_feed, strategy, position_sizer, risk_manager,
                 commission_bps=0, slippage_bps=0, initial_cash=100000):
        self.data_feed = data_feed
        self.strategy = strategy
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.initial_cash = initial_cash

    def run(self):
        df = self.data_feed.load()
        sig = self.strategy.generate_signals(df)
        w = self.position_sizer.target_weights(df, sig)
        w = self.risk_manager.apply(df, w)

        # Compute returns with simple cost model
        ret = df["Close"].pct_change().fillna(0.0)
        w_shift = w.shift().fillna(0.0)  # hold previous day's weight
        gross = w_shift * ret

        # turnover & costs
        tw = w_shift
        tw_next = w.fillna(0.0)
        turnover = (tw_next - tw).abs()
        cost = turnover * (self.commission_bps + self.slippage_bps) / 1e4

        net = gross - cost.fillna(0.0)
        equity = (1 + net).cumprod() * self.initial_cash
        equity.name = "Equity"
        trades = pd.DataFrame({"signal": sig, "weight": w, "turnover": turnover, "ret": ret})
        return equity, trades
