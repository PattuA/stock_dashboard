from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# -------------------------------
# Math helpers (Black-Scholes)
# -------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _put_delta_bs(spot: float, strike: float, t_years: float, r: float, iv: float) -> float:
    """Black-Scholes put delta (negative in [-1, 0])."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return float("nan")
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    return _norm_cdf(d1) - 1.0


# -------------------------------
# Data helpers
# -------------------------------
@st.cache_data(ttl=60 * 5, show_spinner=False)
def _load_hist_close(symbol: str, lookback_days: int = 120) -> pd.Series:
    df = yf.download(symbol, period=f"{lookback_days}d", progress=False, auto_adjust=True)
    return df["Close"].dropna() if not df.empty else pd.Series(dtype=float)


def _atr_percent(symbol: str, lookback: int = 14) -> float:
    df = yf.download(symbol, period="6mo", progress=False, auto_adjust=False)
    if df.empty:
        return float("nan")
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(lookback, min_periods=lookback).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    return float(atr / price * 100.0) if pd.notna(atr) and price > 0 else float("nan")


def _next_earnings(symbol: str) -> Optional[pd.Timestamp]:
    """Best-effort look-up of the next earnings date."""
    try:
        tkr = yf.Ticker(symbol)
        data = tkr.get_earnings_dates(limit=1)
        if isinstance(data, pd.DataFrame) and not data.empty:
            return pd.Timestamp(data.index[-1].to_pydatetime())
    except Exception:
        pass
    try:
        cal = yf.Ticker(symbol).calendar
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            return pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
    except Exception:
        pass
    return None


def _pick_expiration(expirations: List[str], target_dte: int) -> Optional[str]:
    if not expirations:
        return None
    today = datetime.now(timezone.utc).date()
    best, best_diff = None, float("inf")
    for expiry in expirations:
        try:
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        diff = abs(dte - target_dte)
        if diff < best_diff:
            best, best_diff = expiry, diff
    return best


def _pick_put_by_delta(
    chain_puts: pd.DataFrame,
    spot: float,
    expiration_str: str,
    target_delta: float,
    r_annual: float = 0.02,
) -> Optional[pd.Series]:
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
        oi = float(row.get("openInterest", np.nan))
        vol = float(row.get("volume", np.nan))
        if not (np.isfinite(strike) and np.isfinite(iv) and iv > 0 and np.isfinite(bid) and np.isfinite(ask)):
            continue
        delta = abs(_put_delta_bs(spot, strike, t_years, r_annual, iv))
        if not np.isfinite(delta):
            continue
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else np.nan
        spread_pct = (ask - bid) / ask * 100.0 if ask > 0 else np.nan
        rows.append(
            {
                "strike": strike,
                "iv": iv,
                "delta": delta,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread_pct,
                "oi": oi,
                "vol": vol,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["delta_diff"] = (df["delta"] - target_delta).abs()
    return df.sort_values(["delta_diff", "spread_pct"]).iloc[0]


def _score_row(premium_yield: float, spread_pct: float, atr_pct: float, days_to_earnings: Optional[int]) -> float:
    """Composite score: higher is better for CSP selling."""
    premium_component = np.clip(premium_yield, 0, 0.2) / 0.2
    spread_component = 1.0 - np.clip(spread_pct, 0, 8) / 8.0
    risk_component = 1.0 - np.clip(atr_pct, 0, 6) / 6.0
    earnings_component = 1.0
    if days_to_earnings is not None:
        if days_to_earnings <= 5:
            earnings_component = 0.3
        elif days_to_earnings <= 10:
            earnings_component = 0.6
        elif days_to_earnings <= 20:
            earnings_component = 0.8
    return float(100 * (0.45 * premium_component + 0.20 * spread_component + 0.25 * risk_component + 0.10 * earnings_component))


# -------------------------------
# Public UI
# -------------------------------
def render_csp_scanner() -> None:
    st.subheader("CSP Scanner (Real-Time)")

    with st.expander("Scanner settings", expanded=True):
        watchlist = st.text_input(
            "Tickers (comma-separated)",
            value=(
                "SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMD, TSLA, META, GOOGL, AMZN, "
                "JPM, BAC, XOM, KO, PEP, WMT, COST, ORCL, CRM"
            ),
            key="csp_scan_watchlist",
        )
        target_dte = st.slider("Target DTE (days)", 7, 60, 30, step=1, key="csp_scan_dte")
        target_delta = st.slider("Target put delta", 0.05, 0.40, 0.20, step=0.01, key="csp_scan_delta")
        min_oi = st.number_input("Minimum open interest", min_value=0, value=50, step=10, key="csp_scan_min_oi")
        max_spread_pct = st.slider("Max bid-ask spread %", 0.0, 10.0, 6.0, step=0.5, key="csp_scan_spread")
        exclude_earnings_days = st.slider("Exclude if earnings within (days)", 0, 30, 7, step=1, key="csp_scan_excl_earn")

    tickers = [ticker.strip().upper() for ticker in watchlist.split(",") if ticker.strip()]
    total = len(tickers)
    progress = st.progress(0.0) if total else None

    rows = []
    for index, symbol in enumerate(tickers, start=1):
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", {}) or {}
            spot = float(fast_info.get("lastPrice", float("nan")))
            if not np.isfinite(spot) or spot <= 0:
                history = _load_hist_close(symbol, 5)
                spot = float(history.iloc[-1]) if not history.empty else float("nan")
            expirations = list(getattr(ticker, "options", []))
            expiry = _pick_expiration(expirations, target_dte)
            if not expiry or not np.isfinite(spot):
                continue

            chain = ticker.option_chain(expiry)
            puts = getattr(chain, "puts", None)
            pick = _pick_put_by_delta(puts, spot, expiry, target_delta)
            if pick is None:
                continue

            if min_oi and pick["oi"] < min_oi:
                continue
            if np.isfinite(pick["spread_pct"]) and pick["spread_pct"] > max_spread_pct:
                continue

            premium_yield = (
                float(pick["mid"]) / float(pick["strike"])
                if np.isfinite(pick["mid"]) and float(pick["strike"]) > 0
                else float("nan")
            )
            atr_pct = _atr_percent(symbol)
            earnings_date = _next_earnings(symbol)
            days_to_earnings = None
            if earnings_date is not None:
                days_to_earnings = (earnings_date.date() - datetime.now().date()).days
                if exclude_earnings_days and days_to_earnings is not None and days_to_earnings <= exclude_earnings_days:
                    continue

            score = _score_row(premium_yield, float(pick["spread_pct"]), atr_pct, days_to_earnings)

            rows.append(
                {
                    "Ticker": symbol,
                    "Price": round(spot, 2),
                    "Expiry": expiry,
                    "Strike": round(float(pick["strike"]), 2),
                    "|Delta|": round(float(pick["delta"]), 2),
                    "% OTM": round((1 - float(pick["strike"]) / spot) * 100.0, 2),
                    "Mid": round(float(pick["mid"]), 2) if np.isfinite(pick["mid"]) else np.nan,
                    "Bid": round(float(pick["bid"]), 2),
                    "Ask": round(float(pick["ask"]), 2),
                    "Spread %": round(float(pick["spread_pct"]), 2) if np.isfinite(pick["spread_pct"]) else np.nan,
                    "IV %": round(float(pick["iv"]) * 100.0, 1),
                    "OI": int(pick["oi"]) if np.isfinite(pick["oi"]) else 0,
                    "Vol": int(pick["vol"]) if np.isfinite(pick["vol"]) else 0,
                    "Prem/Strike %": round(premium_yield * 100.0, 2) if np.isfinite(premium_yield) else np.nan,
                    "ATR(14) %": round(atr_pct, 2) if np.isfinite(atr_pct) else np.nan,
                    "Earnings (days)": days_to_earnings if days_to_earnings is not None else "",
                    "Score": round(score, 1),
                }
            )
        except Exception:
            continue
        finally:
            if progress is not None:
                progress.progress(min(index / total, 1.0))

    if progress is not None:
        progress.progress(1.0)

    if not rows:
        st.info("No candidates matched your filters. Relax the thresholds or expand the watchlist.")
        return

    df = pd.DataFrame(rows).sort_values(["Score", "Prem/Strike %"], ascending=[False, False]).reset_index(drop=True)
    st.dataframe(df, width="stretch", hide_index=True)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "Download results (CSV)",
        data=csv_bytes,
        file_name="csp_scanner.csv",
        mime="text/csv",
    )
