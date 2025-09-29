"""
Generic metric utilities: YoY, trend slopes, breadth slopes, etc.
"""

import numpy as np, pandas as pd

def pct_change_yoy(s: pd.Series) -> float:
    """YoY % change from last value vs value 12 months ago."""
    if s.empty: return float("nan")
    s_m = s.resample("ME").last()  # month-end
    if len(s_m) < 13: return float("nan")
    last, prev = s_m.iloc[-1], s_m.iloc[-13]
    return (last / prev - 1.0) * 100.0

def trend_slope_percent(series: pd.Series, window_long=200, window_short=50) -> float:
    """(short_MA - long_MA) / long_MA * 100 at latest point."""
    need = max(window_long, window_short)
    if series is None or len(series) < need: return float("nan")
    sma_l = series.rolling(window_long, min_periods=window_long).mean()
    sma_s = series.rolling(window_short, min_periods=window_short).mean()
    if sma_l.dropna().empty or sma_s.dropna().empty: return float("nan")
    last_l = sma_l.dropna().iloc[-1].item()
    last_s = sma_s.dropna().iloc[-1].item()
    if not np.isfinite(last_l) or last_l == 0.0: return float("nan")
    return (last_s - last_l) / last_l * 100.0

def slope_of_ratio(series: pd.Series, short=50, long=200) -> float:
    """Slope % for a ratio (e.g., RSP/SPY) using same method."""
    if series is None or len(series) < max(short, long): return float("nan")
    sma_s = series.rolling(short, min_periods=short).mean()
    sma_l = series.rolling(long,  min_periods=long).mean()
    if sma_l.dropna().empty or sma_s.dropna().empty: return float("nan")
    last_l, last_s = sma_l.dropna().iloc[-1], sma_s.dropna().iloc[-1]
    if pd.isna(last_l) or last_l == 0: return float("nan")
    return (last_s - last_l) / last_l * 100.0
