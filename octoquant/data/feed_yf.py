import yfinance as yf
import pandas as pd

class YFDataFeed:
    def __init__(self, symbol: str, start: str, end: str, interval: str = "1d"):
        self.symbol, self.start, self.end, self.interval = symbol, start, end, interval

    def load(self) -> pd.DataFrame:
        df = yf.download(self.symbol, start=self.start, end=self.end, interval=self.interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data from yfinance for {self.symbol}")
        df = df.rename(columns=str.title)[["Open","High","Low","Close","Volume"]]
        return df.dropna()
