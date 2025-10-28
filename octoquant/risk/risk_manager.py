import pandas as pd

class BasicRisk:
    def __init__(self, max_gross_leverage=1.0):
        self.max_gross_leverage = max_gross_leverage

    def apply(self, df: pd.DataFrame, w: pd.Series) -> pd.Series:
        return w.clip(-self.max_gross_leverage, self.max_gross_leverage)
