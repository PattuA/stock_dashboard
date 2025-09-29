# compute/csp_attractiveness.py
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf

from compute.metrics import slope_of_ratio
from loaders.yf_loader import load_yf_close, load_breadth_series, load_vix_term

def _safe_last_price(symbol: str) -> float:
    """Robust last price with fallbacks; avoids fast_info KeyError."""
    try:
        tkr = yf.Ticker(symbol)
        lp = None
        try:
            fi = tkr.fast_info  # this can raise in some responses
            if isinstance(fi, dict) and "lastPrice" in fi:
                lp = fi["lastPrice"]
        except Exception:
            lp = None

        if lp is None or not np.isfinite(lp):
            # fallback to history (try faster .history first, then download)
            try:
                hist = tkr.history(period="5d", auto_adjust=True)
            except Exception:
                hist = pd.DataFrame()
            if hist is None or hist.empty:
                df = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
            else:
                df = hist
            if not df.empty:
                return float(df["Close"].dropna().iloc[-1])
            return float("nan")
        return float(lp)
    except Exception:
        df = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
        return float(df["Close"].dropna().iloc[-1]) if not df.empty else float("nan")

def _last_scalar(s):
    """Return the last non-NaN value in a Series as a float, or NaN."""
    try:
        if s is None:
            return float("nan")
        s = s.dropna()
        if s.empty:
            return float("nan")
        val = s.iloc[-1]
        # pandas may give a 0-dim Series/ndarray; coerce to float robustly
        return float(np.asarray(val).item() if hasattr(np.asarray(val), "item") else np.asarray(val))
    except Exception:
        return float("nan")

# --------------------------
# Basic math / BS helpers
# --------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def put_delta_bs(spot: float, strike: float, t_years: float, r: float, iv: float) -> float:
    """Black–Scholes put delta (negative in [-1,0])."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return float("nan")
    d1 = (math.log(spot/strike) + (r + 0.5*iv*iv)*t_years) / (iv*math.sqrt(t_years))
    return _norm_cdf(d1) - 1.0

# --------------------------
# Data helpers
# --------------------------
def pick_expiration(expirations: list[str], target_dte: int) -> Optional[str]:
    if not expirations:
        return None
    today = datetime.now(timezone.utc).date()
    best, best_diff = None, 10**9
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

def atr_percent(symbol: str, lookback: int = 14) -> float:
    """
    Return ATR(lookback) as a % of last close.
    Robust to yfinance quirks: always coerces to scalar floats.
    """
    df = yf.download(symbol, period="6mo", progress=False, auto_adjust=False)
    if df is None or df.empty or not {"High", "Low", "Close"} <= set(df.columns):
        return float("nan")

    # Coerce numeric and drop NaNs
    high  = pd.to_numeric(df["High"], errors="coerce")
    low   = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1).dropna()

    if tr.size < lookback or close.dropna().empty:
        return float("nan")

    atr_val = tr.rolling(lookback, min_periods=lookback).mean().iloc[-1]
    last_px = close.dropna().iloc[-1]

    # Force true scalars (not 0-dim Series/arrays)
    atr_val = float(np.asarray(atr_val))
    last_px = float(np.asarray(last_px))

    if not np.isfinite(atr_val) or not np.isfinite(last_px) or last_px <= 0:
        return float("nan")

    return float(atr_val / last_px * 100.0)


def ma200_distance_pct(symbol: str) -> float:
    s = load_yf_close(symbol, period="2y", auto_adjust=True)  # after you added 'period' support
    if s is None or s.empty:
        return float("nan")

    ma200 = s.rolling(200, min_periods=200).mean()
    if ma200.dropna().empty:
        return float("nan")

    last = _last_scalar(s)
    mval = _last_scalar(ma200)

    if not np.isfinite(last) or not np.isfinite(mval) or mval == 0:
        return float("nan")

    return (last / mval - 1.0) * 100.0


def atm_iv_for_expiry(puts: pd.DataFrame, calls: pd.DataFrame, spot: float) -> float:
    """
    Best-effort ATM IV: pick the option (put or call) whose strike is closest to spot
    and return its impliedVolatility.
    """
    best_iv = float("nan")
    best_diff = 10**9
    for df in [puts, calls]:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            k = float(row.get("strike", np.nan))
            iv = float(row.get("impliedVolatility", np.nan))
            if not (np.isfinite(k) and np.isfinite(iv) and iv > 0):
                continue
            diff = abs(k - spot)
            if diff < best_diff:
                best_diff, best_iv = diff, iv
    return best_iv  # fraction, e.g., 0.25 = 25%

def pick_put_by_delta(puts: pd.DataFrame, spot: float, expiry: str,
                      target_delta: float, r_annual: float = 0.02) -> Optional[pd.Series]:
    """
    Pick the put whose |BS delta| is closest to target_delta.
    """
    if puts is None or puts.empty or spot <= 0:
        return None
    try:
        exp = datetime.strptime(expiry, "%Y-%m-%d")
        t_years = max(1e-6, (exp - datetime.now()).days / 365.25)
    except Exception:
        return None

    rows = []
    for _, row in puts.iterrows():
        k = float(row.get("strike", np.nan))
        iv = float(row.get("impliedVolatility", np.nan))
        bid = float(row.get("bid", np.nan)); ask = float(row.get("ask", np.nan))
        oi  = float(row.get("openInterest", np.nan)); vol = float(row.get("volume", np.nan))
        if not (np.isfinite(k) and np.isfinite(iv) and iv > 0 and np.isfinite(bid) and np.isfinite(ask)):
            continue
        delta = abs(put_delta_bs(spot, k, t_years, r_annual, iv))
        if not np.isfinite(delta):
            continue
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else np.nan
        spr = (ask - bid) / ask * 100.0 if ask > 0 else np.nan
        rows.append({"strike": k, "iv": iv, "delta": delta, "bid": bid, "ask": ask,
                     "mid": mid, "spread_pct": spr, "oi": oi, "vol": vol})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["delta_diff"] = (df["delta"] - target_delta).abs()
    return df.sort_values(["delta_diff", "spread_pct"]).iloc[0]

# --------------------------
# Main scoring
# --------------------------
def csp_attractiveness(
    symbol: str,
    target_dte: int = 30,
    target_delta: float = 0.20,
    r_annual: float = 0.02,
) -> Dict[str, float | str]:
    """
    Returns a dict with suggested expiry/strike and a 0–100 score broken down by components.
    Components:
      - Premium Richness: ATM IV (normalized)
      - Liquidity: spread% and OI of chosen contract
      - Safety: ATR% (lower better), buffer in ATR multiples, 200d distance
      - Market Regime: VIX term contango and breadth slope (RSP/SPY)
    """
    tkr = yf.Ticker(symbol)
    spot = _safe_last_price(symbol)
    if not np.isfinite(spot) or spot <= 0:
        return {"ok": False, "msg": "Unable to fetch last price"}

    if not np.isfinite(spot) or spot <= 0:
        hist = load_yf_close(symbol, period="5d", auto_adjust=True)
        spot = float(hist.iloc[-1]) if not hist.empty else float("nan")
    expirations = list(getattr(tkr, "options", []))
    expiry = pick_expiration(expirations, target_dte)
    if not expiry or not np.isfinite(spot):
        return {"ok": False, "msg": "No expiry/price available"}

    chain = tkr.option_chain(expiry)
    puts, calls = getattr(chain, "puts", None), getattr(chain, "calls", None)

    pick = pick_put_by_delta(puts, spot, expiry, target_delta, r_annual=r_annual)
    if pick is None:
        return {"ok": False, "msg": "No suitable put found"}

    # Features
    atm_iv = atm_iv_for_expiry(puts, calls, spot)            # 0.2 = 20%
    spread_pct = float(pick["spread_pct"]) if np.isfinite(pick["spread_pct"]) else np.nan
    oi = float(pick["oi"]) if np.isfinite(pick["oi"]) else 0.0
    mid = float(pick["mid"]) if np.isfinite(pick["mid"]) else np.nan
    strike = float(pick["strike"])
    perc_otm = (1.0 - strike / spot) * 100.0

    atr_pct = float(atr_percent(symbol))

    ma200_pct = ma200_distance_pct(symbol)  # + above MA, - below MA
    buffer_atr = perc_otm / atr_pct if (np.isfinite(perc_otm) and np.isfinite(atr_pct) and atr_pct > 0) else np.nan

    # ---------- market regime ----------
    term = load_vix_term()  # expects dict-like: {"vix": Series, "vix3m": Series}

    vix   = _last_scalar(term.get("vix"))
    vix3m = _last_scalar(term.get("vix3m"))
    contango = (vix3m - vix) if (np.isfinite(vix) and np.isfinite(vix3m)) else float("nan")

    breadth = load_breadth_series()  # Series like RSP/SPY (or your ratio)
    if breadth is not None and not breadth.empty:
        bs = slope_of_ratio(breadth)      # should return a scalar
        breadth_slope = float(bs) if np.isfinite(bs) else float("nan")
    else:
        breadth_slope = float("nan")


    # --------------------------
    # Component scores (0..1)
    # --------------------------
    # Premium Richness: normalize IV between 15%..60%
    pr = 0.0 if not np.isfinite(atm_iv) else float(np.clip((atm_iv - 0.15) / (0.60 - 0.15), 0, 1))
    # Liquidity: tight spreads, decent OI
    liq_spread = 1.0 - float(np.clip((spread_pct if np.isfinite(spread_pct) else 10.0) / 8.0, 0, 1))
    liq_oi = float(np.clip((np.log10(max(oi, 1)) - 1) / (4 - 1), 0, 1))  # 10^1..10^4
    liq = 0.7 * liq_spread + 0.3 * liq_oi
    # Safety: more OTM vs ATR, lower ATR%, above MA200
    saf_buffer = float(np.clip((buffer_atr - 1.0) / (3.0 - 1.0), 0, 1))  # 1x..3x ATR
    saf_atr = 1.0 - float(np.clip((atr_pct if np.isfinite(atr_pct) else 6.0) / 6.0, 0, 1))  # <=6% is better
    saf_ma = float(np.clip((ma200_pct + 10.0) / (20.0), 0, 1))  # -10%..+10% around MA200
    saf = 0.5 * saf_buffer + 0.3 * saf_atr + 0.2 * saf_ma
    # Market regime: contango (3M-1M) and breadth slope
    mr_contango = float(np.clip((contango - 0.5) / (4.0 - 0.5), 0, 1))   # 0.5..4.0 pts
    mr_breadth  = float(np.clip((breadth_slope + 0.5) / (1.5), 0, 1))    # -0.5..+1.0 %
    mr = 0.6 * mr_contango + 0.4 * mr_breadth

    # Weights → total score
    total = 100.0 * (0.40 * pr + 0.20 * liq + 0.25 * saf + 0.15 * mr)

    stance = (
        "Aggressive" if total >= 75 else
        "Moderate"   if total >= 55 else
        "Defensive"
    )

    return {
        "ok": True,
        "symbol": symbol,
        "price": round(spot, 2),
        "expiry": expiry,
        "strike": round(strike, 2),
        "delta": round(float(pick["delta"]), 2),
        "mid": round(mid, 2) if np.isfinite(mid) else np.nan,
        "%OTM": round(perc_otm, 2) if np.isfinite(perc_otm) else np.nan,
        "IV_atm_%": round(atm_iv * 100.0, 1) if np.isfinite(atm_iv) else np.nan,
        "spread_%": round(spread_pct, 2) if np.isfinite(spread_pct) else np.nan,
        "OI": int(oi),
        "ATR_%": round(atr_pct, 2) if np.isfinite(atr_pct) else np.nan,
        "buffer_ATR_x": round(buffer_atr, 2) if np.isfinite(buffer_atr) else np.nan,
        "MA200_%": round(ma200_pct, 2) if np.isfinite(ma200_pct) else np.nan,
        "contango": round(contango, 2) if np.isfinite(contango) else np.nan,
        "breadth_slope_%": round(breadth_slope, 2) if np.isfinite(breadth_slope) else np.nan,
        # component scores
        "score_total": round(total, 1),
        "score_premium": round(100 * pr, 1),
        "score_liquidity": round(100 * liq, 1),
        "score_safety": round(100 * saf, 1),
        "score_regime": round(100 * mr, 1),
        "stance": stance,
    }
