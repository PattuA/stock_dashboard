# panels/csp_scanner_panel.py
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import START_EQ_DATE

# -------------------------------
# Math helpers (Black–Scholes)
# -------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _put_delta_bs(spot: float, strike: float, t_years: float, r: float, iv: float) -> float:
    """Black–Scholes delta (put). Returns negative number in [-1, 0]."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return float("nan")
    d1 = (math.log(spot/strike) + (r + 0.5*iv*iv)*t_years) / (iv*math.sqrt(t_years))
    # put delta = N(d1) - 1
    return _norm_cdf(d1) - 1.0

# -------------------------------
# Data helpers
# -------------------------------
@st.cache_data(ttl=60*5, show_spinner=False)
def _load_hist_close(symbol: str, lookback_days: int = 120) -> pd.Series:
    df = yf.download(symbol, period=f"{lookback_days}d", progress=False, auto_adjust=True)
    return df["Close"].dropna() if not df.empty else pd.Series(dtype=float)

def _atr_percent(symbol: str, lookback: int = 14) -> float:
    df = yf.download(symbol, period="6mo", progress=False, auto_adjust=False)
    if df.empty:
        return float("nan")
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=lookback).mean().iloc[-1]
    price = close.iloc[-1]
    return float(atr / price * 100.0) if pd.notna(atr) and price > 0 else float("nan")

def _next_earnings(symbol: str) -> Optional[pd.Timestamp]:
    """Best-effort: try multiple yf fields; gracefully return None if unavailable."""
    try:
        tkr = yf.Ticker(symbol)
        # Try new API first
        edf = tkr.get_earnings_dates(limit=1)
        if isinstance(edf, pd.DataFrame) and not edf.empty:
            ts = edf.index[-1].to_pydatetime()
            return pd.Timestamp(ts)
    except Exception:
        pass
    try:
        cal = yf.Ticker(symbol).calendar  # legacy
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"].iloc[0]
            return pd.to_datetime(val)
    except Exception:
        pass
    return None

def _pick_expiration(expirations: List[str], target_dte: int) -> Optional[str]:
    """Pick the expiration closest to target_dte (in calendar days)."""
    if not expirations:
        return None
    today = datetime.now(timezone.utc).date()
    best = None
    best_diff = 10**9
    for e in expirations:
        try:
            ed = datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (ed - today).days
        diff = abs(dte - target_dte)
        if diff < best_diff:
            best, best_diff = e, diff
    return best

def _pick_put_by_delta(
    chain_puts: pd.DataFrame,
    spot: float,
    expiration_str: str,
    target_delta: float,
    r_annual: float = 0.02,
) -> Optional[pd.Series]:
    """
    From a puts chain (one expiry), pick the contract whose absolute BS-delta
    is closest to target_delta (e.g., 0.20). Uses each row's impliedVolatility.
    """
    if chain_puts is None or chain_puts.empty or spot <= 0:
        return None
    try:
        exp = datetime.strptime(expiration_str, "%Y-%m-%d")
        t_years = max(1e-6, (exp - datetime.now()).days / 365.25)
    except Exception:
        return None

    rows = []
    for _, row in chain_puts.iterrows():
        strike = float(row.get("strike", np.nan))
        iv = float(row.get("impliedVolatility", np.nan))
        bid = float(row.get("bid", np.nan))
        ask = float(row.get("ask", np.nan))
        oi  = float(row.get("openInterest", np.nan))
        vol = float(row.get("volume", np.nan))
        if not (np.isfinite(strike) and np.isfinite(iv) and iv > 0 and np.isfinite(bid) and np.isfinite(ask)):
            continue
        delta = abs(_put_delta_bs(spot, strike, t_years, r_annual, iv))
        if not np.isfinite(delta):
            continue
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else np.nan
        spread_pct = (ask - bid) / ask * 100.0 if ask > 0 else np.nan
        rows.append({
            "strike": strike, "iv": iv, "delta": delta, "bid": bid, "ask": ask,
            "mid": mid, "spread_pct": spread_pct, "oi": oi, "vol": vol
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["delta_diff"] = (df["delta"] - target_delta).abs()
    pick = df.sort_values(["delta_diff", "spread_pct"]).iloc[0]
    return pick

def _score_row(premium_yield: float, spread_pct: float, atr_pct: float, days_to_earnings: Optional[int]) -> float:
    """
    Simple composite score: higher is better for CSP selling.
    Penalize wide spreads, high ATR%, and earnings proximity.
    """
    # Normalize inputs
    py = np.clip(premium_yield, 0, 0.2) / 0.2       # cap at 20%/yr-ish per contract
    sp = 1.0 - np.clip(spread_pct, 0, 8) / 8.0       # 0..1 (0 = 8% spread)
    risk = 1.0 - np.clip(atr_pct, 0, 6) / 6.0        # 0..1 (6% ATR cap)
    earn_pen = 1.0
    if days_to_earnings is not None:
        if days_to_earnings <= 5:   earn_pen = 0.3
        elif days_to_earnings <= 10: earn_pen = 0.6
        elif days_to_earnings <= 20: earn_pen = 0.8
    # weights: yield 45%, liquidity 20%, risk 25%, earnings 10%
    return float(100 * (0.45*py + 0.20*sp + 0.25*risk + 0.10*earn_pen))

# -------------------------------
# Public UI
# -------------------------------
def render_csp_scanner():
    st.subheader("🧮 CSP Scanner (Real-Time)")

    # Controls
    with st.expander("Scanner settings", expanded=True):
        watchlist = st.text_input(
            "Tickers (comma-separated)",
            value="SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMD, TSLA, META, GOOGL, AMZN, JPM, BAC, XOM, KO, PEP, WMT, COST, ORCL, CRM",
        )
        target_dte = st.slider("Target DTE (days)", 7, 60, 30, step=1)
        target_delta = st.slider("Target put delta", 0.05, 0.40, 0.20, step=0.01)
        min_oi = st.number_input("Min open interest (filter)", min_value=0, value=50, step=10)
        max_spread_pct = st.slider("Max bid–ask spread %", 0.0, 10.0, 6.0, step=0.5)
        exclude_earnings_days = st.slider("Exclude if earnings within (days)", 0, 30, 7, step=1)

    tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]

    rows = []
    progress = st.progress(0.0)
    for i, sym in enumerate(tickers, 1):
        try:
            tkr = yf.Ticker(sym)
            spot = float(tkr.fast_info["lastPrice"]) if "lastPrice" in tkr.fast_info else float("nan")
            if not np.isfinite(spot) or spot <= 0:
                # fallback to history
                hist = _load_hist_close(sym, 5)
                spot = float(hist.iloc[-1]) if not hist.empty else float("nan")
            exps = list(tkr.options) if hasattr(tkr, "options") else []
            exp = _pick_expiration(exps, target_dte)
            if not exp or not np.isfinite(spot):
                progress.progress(i/len(tickers))
                continue

            chain = tkr.option_chain(exp)
            puts = getattr(chain, "puts", None)
            pick = _pick_put_by_delta(puts, spot, exp, target_delta)
            if pick is None:
                progress.progress(i/len(tickers))
                continue

            # filters
            if min_oi and pick["oi"] < min_oi:
                progress.progress(i/len(tickers)); continue
            if np.isfinite(pick["spread_pct"]) and pick["spread_pct"] > max_spread_pct:
                progress.progress(i/len(tickers)); continue

            # derived metrics
            premium_yield = float(pick["mid"]) / float(pick["strike"]) if np.isfinite(pick["mid"]) and pick["strike"] > 0 else float("nan")
            atr_pct = _atr_percent(sym)
            e = _next_earnings(sym)
            days_to_earnings = None
            if e is not None:
                days_to_earnings = (e.date() - datetime.now().date()).days

            score = _score_row(premium_yield, float(pick["spread_pct"]), atr_pct, days_to_earnings)

            rows.append({
                "Ticker": sym,
                "Price": round(spot, 2),
                "Expiry": exp,
                "Strike": round(float(pick["strike"]), 2),
                "Δ (abs)": round(float(pick["delta"]), 2),
                "% OTM": round((1 - float(pick["strike"])/spot)*100.0, 2),
                "Mid": round(float(pick["mid"]), 2) if np.isfinite(pick["mid"]) else np.nan,
                "Bid": round(float(pick["bid"]), 2),
                "Ask": round(float(pick["ask"]), 2),
                "Spread %": round(float(pick["spread_pct"]), 2) if np.isfinite(pick["spread_pct"]) else np.nan,
                "IV": round(float(pick["iv"])*100.0, 1),
                "OI": int(pick["oi"]),
                "Vol": int(pick["vol"]),
                "Prem/Strike %": round(premium_yield*100.0, 2) if np.isfinite(premium_yield) else np.nan,
                "ATR(14) %": round(atr_pct, 2) if np.isfinite(atr_pct) else np.nan,
                "Earnings (days)": days_to_earnings if days_to_earnings is not None else "",
                "Score": round(score, 1),
            })
        except Exception:
            # keep going if any symbol fails
            pass
        finally:
            progress.progress(i/len(tickers))

    if not rows:
        st.info("No candidates matched your filters. Try relaxing OI / spread filters or expand the watchlist.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["Score", "Prem/Strike %"], ascending=[False, False]).reset_index(drop=True)

    st.dataframe(df, width="stretch", hide_index=True)

    # CSV download
    csv = df.to_csv(index=False).encode()
    st.download_button("Download results (CSV)", data=csv, file_name="csp_scanner.csv", mime="text/csv")
