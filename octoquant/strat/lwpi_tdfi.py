import pandas as pd
import numpy as np
from octoquant.strat.base import Strategy

class LwpiTdfiStrategy(Strategy):
    """
    Strategy based on Larry Williams Proxy Index (LWPI) and Trend Direction Force Index v2 (TDFI).
    Includes stateful logic for ATR-based or Fixed Stop Loss/Take Profit.
    """
    def __init__(self, lwpi_period=135, lwpi_smooth=8,
                 tdfi_lookback=21, tdfi_mma=5, tdfi_smma=15, tdfi_n=5,
                 tp_pct=0.03, sl_pct=0.03, 
                 sl_type="fixed", atr_period=14, atr_mult=3.0):
        self.lwpi_period = lwpi_period
        self.lwpi_smooth = lwpi_smooth
        self.tdfi_lookback = tdfi_lookback
        self.tdfi_mma = tdfi_mma
        self.tdfi_smma = tdfi_smma
        self.tdfi_n = tdfi_n
        
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.sl_type = sl_type  # "fixed" or "atr"
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators and attach to dataframe."""
        df = df.copy()
        
        # --- ATR Calculation (if needed) ---
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(self.atr_period).mean()

        # --- LWPI Implementation ---
        high_roll = high.rolling(self.lwpi_period).max()
        low_roll = low.rolling(self.lwpi_period).min()
        
        denom = (high_roll - low_roll).replace(0, np.nan)
        rsv = 100 * (close - low_roll) / denom
        df["LWPI"] = rsv.rolling(self.lwpi_smooth).mean()
        
        # --- TDFI v2 Implementation ---
        change = close - prev_close
        force = change * df["Volume"]
        abs_force = change.abs() * df["Volume"]
        
        s_force = force.ewm(span=self.tdfi_lookback).mean()
        s_abs_force = abs_force.ewm(span=self.tdfi_lookback).mean()
        
        tdfi_raw = s_force / s_abs_force.replace(0, 1)
        tdfi_smooth = tdfi_raw.ewm(span=self.tdfi_mma).mean()
        tdfi_final = tdfi_smooth.ewm(span=self.tdfi_smma).mean()
        
        df["TDFI"] = tdfi_final
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self.calc_indicators(df)
        
        # Vectorized Entry Conditions
        long_cond = (df["LWPI"] < 50) & (df["TDFI"] > 0.05)
        short_cond = (df["LWPI"] > 50) & (df["TDFI"] < -0.05)
        
        # State Machine loop for Exit Logic (SL/TP)
        signals = np.zeros(len(df))
        
        position = 0          # 0, 1, -1
        entry_price = 0.0
        stop_price = 0.0
        take_price = 0.0
        
        # Pre-convert to numpy for speed
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        atrs = df["ATR"].values
        l_cond = long_cond.values
        s_cond = short_cond.values
        
        for i in range(1, len(df)):
            # Default: maintain previous position
            signals[i] = position
            
            curr_close = closes[i]
            curr_high = highs[i]
            curr_low = lows[i]
            curr_atr = atrs[i] if not np.isnan(atrs[i]) else 0.0
            
            # 1. Check Exits if in position
            if position != 0:
                hit_sl = False
                hit_tp = False
                
                if position == 1: # LONG
                    if curr_low <= stop_price: hit_sl = True
                    if curr_high >= take_price: hit_tp = True
                elif position == -1: # SHORT
                    if curr_high >= stop_price: hit_sl = True
                    if curr_low <= take_price: hit_tp = True
                    
                if hit_sl or hit_tp:
                    position = 0
                    signals[i] = 0
                    entry_price = 0.0
                    continue # Exit executed, allow re-entry next bar? Usually no, wait next signal.
            
            # 2. Check Entries (only if flat or reversing - reversing covered by separate exits usually)
            # Strategy says: "Opposite signals should close the existing position before opening a new one"
            # This loop treats "Exit" then "Entry" in same bar is hard.
            # We will prioritize: Check Exit -> If Flat -> Check Entry.
            # If Reverse Signal -> Immediate Flip (handled below).
            
            new_signal = 0
            if l_cond[i]: new_signal = 1
            if s_cond[i]: new_signal = -1
            
            if new_signal != 0:
                if position != 0 and position != new_signal:
                    # Reverse
                    position = new_signal
                    signals[i] = new_signal
                    entry_price = curr_close # Approximation: Fill at Close
                    
                    # Set SL/TP
                    if self.sl_type == "atr":
                         sl_dist = curr_atr * self.atr_mult
                         tp_dist = curr_atr * self.atr_mult # Using same R:R? User asked for 3% fixed previously.
                         # User said: "Consider using ATR-based stops". Didn't specify TP.
                         # We'll assume TP remains Fixed 3% OR ATR based.
                         # Let's use Fixed TP for now as requested "Stop Loss Adjustment", not TP.
                         # Or maybe ATR R:R 1:1?
                         # Let's keep TP Fixed 3% as per original request, only change SL to ATR.
                         
                         tp_dist = entry_price * self.tp_pct # Fixed TP
                         
                    else:
                         sl_dist = entry_price * self.sl_pct
                         tp_dist = entry_price * self.tp_pct
                    
                    if position == 1:
                        if self.sl_type == "atr":
                            stop_price = entry_price - sl_dist
                            take_price = entry_price * (1 + self.tp_pct) 
                        else:
                            stop_price = entry_price * (1 - self.sl_pct)
                            take_price = entry_price * (1 + self.tp_pct)
                            
                    else: # Short
                        if self.sl_type == "atr":
                            stop_price = entry_price + sl_dist
                            take_price = entry_price * (1 - self.tp_pct)
                        else:
                            stop_price = entry_price * (1 + self.sl_pct)
                            take_price = entry_price * (1 - self.tp_pct)

                elif position == 0:
                    # Open New
                    position = new_signal
                    signals[i] = new_signal
                    entry_price = curr_close
                    
                    if self.sl_type == "atr":
                         sl_dist = curr_atr * self.atr_mult
                    else:
                         sl_dist = entry_price * self.sl_pct
                    
                    if position == 1:
                        stop_price = entry_price - sl_dist if self.sl_type == "atr" else entry_price * (1 - self.sl_pct)
                        take_price = entry_price * (1 + self.tp_pct)
                    else:
                        stop_price = entry_price + sl_dist if self.sl_type == "atr" else entry_price * (1 + self.sl_pct)
                        take_price = entry_price * (1 - self.tp_pct)
        
        return pd.Series(signals, index=df.index)
