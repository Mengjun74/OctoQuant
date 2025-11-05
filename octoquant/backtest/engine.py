import numpy as np
import pandas as pd
from typing import Dict, Optional


class BacktestEngine:
    """
    单标的、日线优先的执行引擎（含小资金友好特性）：
    - 执行时点可配置（避免前视偏差）
    - 单根最大调仓限制（抑制手续费与噪音再平衡）
    - 最小名义金额过滤（小资金避免“下不了单”的虚假换手）
    成本模型仍为：cost = effective_turnover * (commission_bps + slippage_bps) / 1e4
    """

    def __init__(
        self,
        data_feed,
        strategy,
        position_sizer,
        risk_manager,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        initial_cash: float = 100000.0,
        execution_cfg: Optional[Dict] = None,
    ):
        self.data_feed = data_feed
        self.strategy = strategy
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.initial_cash = float(initial_cash)

        self.execution = {
            "exec_mode": "close_to_next_open",  # close_to_next_open | close_to_close | vwap_bar
            "max_turnover_per_bar": 1.0,        # 每根bar最大调仓幅度（0~1）
            "min_notional": 0.0,                # 本币最小名义金额（如 10 USDT）
            "lot_rounding": "none",             # none | floor | round
        }
        if execution_cfg:
            self.execution.update(execution_cfg)

    # ---------- 内部工具 ----------

    @staticmethod
    def _calc_returns(df: pd.DataFrame, exec_mode: str) -> pd.Series:
        """
        根据执行时点生成每根bar的“可获得收益率序列”：
        - close_to_next_open: ret_t = (Open_{t+1} / Close_{t} - 1)
        - close_to_close:     ret_t = (Close_t / Close_{t-1} - 1)
        - vwap_bar:           需要 df['VWAP']，否则回退到 close_to_close
        """
        if exec_mode == "close_to_next_open" and "Open" in df.columns:
            ret = (df["Open"].shift(-1) / df["Close"] - 1.0)
        elif exec_mode == "vwap_bar" and "VWAP" in df.columns:
            ret = df["VWAP"].pct_change()
        else:
            # 默认/回退：收盘对收盘
            ret = df["Close"].pct_change()
        return ret.fillna(0.0)

    @staticmethod
    def _apply_lot_rounding(delta_w: float, mode: str) -> float:
        """
        对权重变化做近似“下单单位”取整（没有最小手数量时，用保守近似）
        - none: 不处理
        - floor: 向下取整到 2 位小数（更保守；防止小额超下限）
        - round: 四舍五入到 2 位小数
        """
        if mode == "none":
            return delta_w
        # 以两位小数为“名义单位”近似（小账户更保守）；如需更精细可接交易所的最小手数
        scale = 100.0
        if mode == "floor":
            return np.floor(delta_w * scale) / scale
        elif mode == "round":
            return np.round(delta_w * scale) / scale
        return delta_w

    # ---------- 主流程 ----------

    def run(self):
        """
        逐期迭代（而非全向量），以便在每一步使用“上一期的 equity”
        来判断 min_notional 与换手上限，从而符合小资金实盘约束。
        返回：
            equity: pd.Series，账户净值
            trades: pd.DataFrame，包含目标权重、有效权重、换手、毛/净收益等
        """
        df = self.data_feed.load().copy()
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy() if "Open" in df.columns else df

        # 1) 策略&仓位
        signal = self.strategy.generate_signals(df).astype(float)
        w_target = self.position_sizer.target_weights(df, signal).astype(float)
        w_target = self.risk_manager.apply(df, w_target).astype(float)  # 杠杆裁剪等

        # 2) 执行配置
        exec_mode = str(self.execution.get("exec_mode", "close_to_next_open"))
        max_turnover = float(self.execution.get("max_turnover_per_bar", 1.0))
        min_notional = float(self.execution.get("min_notional", 0.0))
        lot_rounding = str(self.execution.get("lot_rounding", "none"))

        # 3) 每期收益（由执行时点决定）
        bar_ret = self._calc_returns(df, exec_mode)  # 与 df.index 对齐

        index = df.index
        n = len(index)
        equity = pd.Series(index=index, dtype="float64")
        equity.iloc[0] = self.initial_cash

        # 有效权重（持仓）序列，用于记录
        w_eff = pd.Series(0.0, index=index, dtype="float64")

        # 交易统计
        eff_turnover = pd.Series(0.0, index=index, dtype="float64")
        cost_series = pd.Series(0.0, index=index, dtype="float64")
        gross_series = pd.Series(0.0, index=index, dtype="float64")
        net_series = pd.Series(0.0, index=index, dtype="float64")

        # 4) 逐期推进
        for i in range(1, n):
            t_prev = index[i - 1]
            t = index[i]

            prev_equity = float(equity.iloc[i - 1])
            prev_w = float(w_eff.iloc[i - 1])
            target_w = float(w_target.iloc[i])

            # 4.1 先做 max_turnover 限制（目标相对上一期有效权重）
            delta = target_w - prev_w
            delta = np.clip(delta, -max_turnover, max_turnover)

            # 4.2 名义金额过滤（小额不调仓）
            # 名义金额 ≈ |Δw| * 账户净值
            trade_notional = abs(delta) * prev_equity
            if trade_notional < min_notional:
                delta = 0.0

            # 4.3 lot rounding 近似
            delta = self._apply_lot_rounding(delta, lot_rounding)

            # 4.4 本期生效权重
            curr_w = np.clip(prev_w + delta, -1.0, 1.0)
            w_eff.iloc[i] = curr_w

            # 4.5 成本与收益
            # 当期有效换手 = |Δw|
            turnover = abs(curr_w - prev_w)
            eff_turnover.iloc[i] = turnover

            cost = turnover * (self.commission_bps + self.slippage_bps) / 1e4
            cost_series.iloc[i] = cost

            gross = curr_w * float(bar_ret.iloc[i])  # 杠杆化后的毛收益
            gross_series.iloc[i] = gross

            net = gross - cost
            net_series.iloc[i] = net

            # 4.6 更新净值
            equity.iloc[i] = prev_equity * (1.0 + net)

        trades = pd.DataFrame(
            {
                "signal": signal.reindex(index),
                "weight_target": w_target.reindex(index),
                "weight_eff": w_eff,
                "turnover": eff_turnover,
                "ret_gross": gross_series,
                "cost": cost_series,
                "ret_net": net_series,
            },
            index=index,
        )

        equity.name = "Equity"
        return equity, trades
