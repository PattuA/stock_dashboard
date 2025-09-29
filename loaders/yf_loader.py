"""
Yahoo Finance loaders.
Returns clean Series (never DataFrames) and sets names safely.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
import yfinance as yf

from config import START_EQ_DATE



@st.cache_data(ttl=60*30, show_spinner=False)
def load_yf_close(
    symbol: str,
    start: str | None = None,
    period: str | None = None,
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.Series:
    """
    Return Yahoo Finance Close series.
    You may specify either `period` (e.g., '6mo','2y') or `start` (YYYY-MM-DD).
    """
    if period:
        df = yf.download(symbol, period=period, auto_adjust=auto_adjust, progress=progress)
    else:
        # default start if not provided
        start = start or "2018-01-01"
        df = yf.download(symbol, start=start, auto_adjust=auto_adjust, progress=progress)

    return df["Close"].dropna() if not df.empty else pd.Series(dtype=float)

@st.cache_data(ttl=60*30, show_spinner=False)
def load_equity(symbol="^GSPC", start=START_EQ_DATE) -> pd.Series:
    return load_yf_close(symbol, start=start, auto_adjust=True)

@st.cache_data(ttl=60*30, show_spinner=False)
def load_vix(start=START_EQ_DATE) -> pd.Series:
    return load_yf_close("^VIX", start=start, auto_adjust=False)

@st.cache_data(ttl=60*30, show_spinner=False)
def load_dxy(start=START_EQ_DATE) -> pd.Series:
    s = load_yf_close("DX-Y.NYB", start=start, auto_adjust=True)
    s.name = "DXY"
    return s

@st.cache_data(ttl=60*30, show_spinner=False)
def load_breadth_series(start=START_EQ_DATE) -> pd.Series:
    """RSP/SPY ratio (equal-weight vs cap-weight) as breadth proxy."""
    from math import isfinite
    rsp = load_yf_close("RSP", start=start, auto_adjust=True)
    spy = load_yf_close("SPY", start=start, auto_adjust=True)
    if rsp.empty or spy.empty: return pd.Series(dtype=float)
    ratio = (rsp / spy).dropna()
    ratio.name = "RSP/SPY"
    return ratio

@st.cache_data(ttl=60*30, show_spinner=False)
def load_vix_term(start=START_EQ_DATE) -> dict:
    """^VIX (1M), ^VIX3M (3M), ^VIX9D (9-day) – coerced to Series."""
    def as_named(sym: str, name: str) -> pd.Series:
        s = load_yf_close(sym, start=start, auto_adjust=False)
        s.name = name
        return s
    return {
        "vix":   as_named("^VIX",   "VIX"),
        "vix3m": as_named("^VIX3M", "VIX3M"),
        "vix9d": as_named("^VIX9D", "VIX9D"),
    }
