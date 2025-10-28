from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Settings:
    market: str
    symbols: list
    interval: str
    start: str
    end: str
    commission_bps: float
    slippage_bps: float
    max_gross_leverage: float
    risk: dict
    position: dict
    strategy: dict
    backtest: dict
