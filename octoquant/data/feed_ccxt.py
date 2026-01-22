import ccxt
import pandas as pd
import time
from datetime import datetime, timezone

class CCXTDataFeed:
    def __init__(self, exchange: str, symbol: str, timeframe: str = "30m", start_date: str = None, end_date: str = None, api_key: str = None, secret: str = None, sandbox: bool = False):
        self.raw_exchange_id = exchange
        try:
            exchange_class = getattr(ccxt, exchange)
        except AttributeError:
            raise ValueError(f"Exchange {exchange} not found in ccxt")
        
        config = {}
        if api_key and secret:
             config = {
                'apiKey': api_key,
                'secret': secret,
            }
        
        self.ex = exchange_class(config)
        
        if sandbox:
            try:
                self.ex.set_sandbox_mode(True)
                print(f"[{exchange}] Sandbox mode enabled")
            except Exception as e:
                print(f"[{exchange}] Warning: Sandbox mode not supported or failed: {e}")

        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date

    def load(self) -> pd.DataFrame:
        if not self.start_date:
            raise ValueError("Start date must be provided for backtesting")
        
        since = self.ex.parse8601(f"{self.start_date}T00:00:00Z")
        if self.end_date:
            end_ts = self.ex.parse8601(f"{self.end_date}T00:00:00Z")
        else:
            end_ts = self.ex.milliseconds()

        all_ohlcv = []
        
        print(f"Fetching {self.symbol} ({self.timeframe}) from {self.start_date} to {self.end_date or 'now'}...")
        
        while since < end_ts:
            try:
                ohlcv = self.ex.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                last_ts = ohlcv[-1][0]
                # Check if we didn't advance (avoid infinite loops if exchange returns same candle)
                if last_ts == since:
                    # Move to next candle manually if stuck
                    since += self.ex.parse_timeframe(self.timeframe) * 1000
                else:
                    since = last_ts + self.ex.parse_timeframe(self.timeframe) * 1000
                
                # Check if we passed the end date
                if since >= end_ts:
                    break
                    
                # Rate limit
                time.sleep(self.ex.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(1) # Backoff
                continue

        if not all_ohlcv:
            raise ValueError(f"No OHLCV from {self.ex.id} for {self.symbol}")
            
        df = pd.DataFrame(all_ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep='first')] # Remove duplicates
        
        # Filter by end_date strictly
        if self.end_date:
             df = df[df.index <= f"{self.end_date} 23:59:59"]

        return df.sort_index()
 