"""
OctoQuant Metrics Utilities

Usage:
    from octoquant.backtest.metrics import compute_metrics, pretty_print

    metrics = compute_metrics(equity, periods_per_year=8760, rf=0.0)
    pretty_print(metrics)
"""

from __future__ import annotations
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _years_between(index: pd.DatetimeIndex) -> float:
    """Calendar-year span between first and last timestamps (fractional years)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return np.nan
    delta = (index[-1] - index[0]).total_seconds()
    return delta / (365.25 * 24 * 3600.0)


def cagr(equity: pd.Series, periods_per_year: Optional[float] = None) -> float:
    """
    Compound Annual Growth Rate.
    If DatetimeIndex is present, prefer calendar time; otherwise fall back to periods_per_year.
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0

    years = _years_between(equity.index) if isinstance(equity.index, pd.DatetimeIndex) else np.nan

    if not np.isnan(years) and years > 0:
        return (end / start) ** (1.0 / years) - 1.0

    # fallback to discrete periods
    if periods_per_year is None or periods_per_year <= 0:
        raise ValueError("periods_per_year must be provided when index is not datetime.")
    n_periods = max(len(equity) - 1, 1)
    return (end / start) ** (periods_per_year / n_periods) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: float) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    return float(returns.std(ddof=0)) * math.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    """
    rf: annual risk-free rate (e.g., 0.02 for 2%)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    mean_ann = float(returns.mean()) * periods_per_year
    vol_ann = annualized_vol(returns, periods_per_year)
    denom = max(vol_ann, 1e-12)
    return (mean_ann - rf) / denom


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna().astype(float)
    if len(equity) == 0:
        return 0.0
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())  # negative number, e.g., -0.25


def calmar_ratio(equity: pd.Series, periods_per_year: Optional[float] = None) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return np.inf
    return cagr(equity, periods_per_year=periods_per_year) / dd


def hit_ratio(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def compute_metrics(
    equity: pd.Series,
    periods_per_year: float,
    rf: float = 0.0,
    returns: Optional[pd.Series] = None,
    trades: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Compute a compact set of commonly used performance metrics.

    Args:
        equity: equity curve in currency units (index aligned with bar times).
        periods_per_year: e.g. 252 (daily stocks), 365 (daily crypto), 8760 (hourly crypto).
        rf: annual risk-free rate.
        returns: optional per-period returns. If None, derived from equity pct_change().
        trades: optional DataFrame with a 'turnover' column to summarize average turnover.

    Returns:
        dict of metrics.
    """
    eq = equity.dropna().astype(float)
    rets = returns.dropna().astype(float) if returns is not None else eq.pct_change().dropna()

    metrics = {
        "CAGR": cagr(eq, periods_per_year=periods_per_year),
        "Annualized Vol": annualized_vol(rets, periods_per_year),
        "Sharpe": sharpe_ratio(rets, periods_per_year, rf=rf),
        "Max Drawdown": max_drawdown(eq),
        "Calmar": calmar_ratio(eq, periods_per_year=periods_per_year),
        "Hit Ratio": hit_ratio(rets),
        "Total Return": float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0,
    }

    if trades is not None and "turnover" in trades.columns:
        metrics["Avg Turnover / Period"] = float(trades["turnover"].dropna().mean())

    return metrics


def pretty_print(metrics: Dict[str, float]) -> None:
    df = pd.DataFrame(metrics, index=["Value"]).T
    # format a few key fields nicely
    pct_cols = ["CAGR", "Annualized Vol", "Max Drawdown", "Total Return", "Hit Ratio", "Avg Turnover / Period"]
    for c in pct_cols:
        if c in df.index:
            # these are stored in index; skip
            continue
    # If df is transposed (metrics on rows), format during display:
    with pd.option_context("display.float_format", lambda v: f"{v:0.6f}"):
        print(df)
