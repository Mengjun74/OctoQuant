import ccxt
import pandas as pd

class CCXTDataFeed:
    def __init__(self, exchange: str, symbol: str, timeframe: str = "1h", limit: int = 1500):
        self.ex = getattr(ccxt, exchange)()
        self.symbol, self.timeframe, self.limit = symbol, timeframe, limit

    def load(self) -> pd.DataFrame:
        ohlcv = self.ex.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=self.limit)
        if not ohlcv:
            raise ValueError(f"No OHLCV from {self.ex.id} for {self.symbol}")
        df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"]).set_index("timestamp")
        df.index = pd.to_datetime(df.index, unit="ms")
        return df.dropna()
