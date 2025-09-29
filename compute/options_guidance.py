"""
Heuristics that translate market inputs → aggressiveness for
Cash-Secured Puts (CSP) and Covered Calls (CC).
Outputs stance, target strike delta, and DTE window.
"""

import pandas as pd

def clamp(x, lo, hi): return max(lo, min(hi, x))

def zshape_for_premium(vix):
    if pd.isna(vix): return 0.5
    if vix < 12:  return 0.1
    if vix < 16:  return 0.4
    if vix < 20:  return 0.8
    if vix < 26:  return 1.0
    if vix < 32:  return 0.6
    if vix < 40:  return 0.3
    return 0.1

def contango_score(vix, vix3m):
    if pd.isna(vix) or pd.isna(vix3m): return 0.5
    spread = (vix3m - vix)
    if spread <= -2: return 0.0
    if spread <= 0:  return 0.3
    if spread <= 2:  return 0.7
    return 1.0

def credit_score(baa_spread):
    if pd.isna(baa_spread): return 0.5
    if baa_spread < 1.8: return 1.0
    if baa_spread < 2.3: return 0.8
    if baa_spread < 3.0: return 0.5
    if baa_spread < 4.0: return 0.2
    return 0.0

def trend_score(trend_pct):
    if pd.isna(trend_pct): return 0.5
    if trend_pct >= 2.0: return 1.0
    if trend_pct >= 0.5: return 0.8
    if trend_pct >= 0.0: return 0.6
    if trend_pct >= -1.0: return 0.3
    return 0.1

def breadth_score(rsp_spy_trend):
    if pd.isna(rsp_spy_trend): return 0.5
    if rsp_spy_trend >= 0.5: return 1.0
    if rsp_spy_trend >= 0.0: return 0.7
    if rsp_spy_trend >= -0.5: return 0.4
    return 0.2

def csp_guidance(vix, vix3m, spy_trend, breadth_trend, baa_spread):
    s = (
        0.30*zshape_for_premium(vix) +
        0.20*contango_score(vix, vix3m) +
        0.25*trend_score(spy_trend) +
        0.15*breadth_score(breadth_trend) +
        0.10*credit_score(baa_spread)
    )
    if s >= 0.75:
        level, delta, dte, note = "Aggressive", 0.30, "7–14 days", "Good premium, stable structure"
    elif s >= 0.50:
        level, delta, dte, note = "Moderate", 0.20, "14–30 days", "Mixed signals"
    else:
        level, delta, dte, note = "Defensive", 0.10, "30–45 days", "Elevated risk / weak structure"
    return {"score": round(s*100), "level": level, "delta": delta, "dte": dte, "note": note}

def cc_guidance(vix, vix3m, spy_trend, breadth_trend, baa_spread):
    s = clamp(
        0.30*max(zshape_for_premium(vix), 0.5) +  # richer premium helps calls too
        0.20*(1 - contango_score(vix, vix3m)) +
        0.30*(1 - trend_score(spy_trend)) +
        0.10*(1 - breadth_score(breadth_trend)) +
        0.10*(1 - credit_score(baa_spread)),
        0, 1
    )
    if s >= 0.75:
        level, delta, dte, note = "Aggressive", 0.35, "7–14 days", "Choppy/weak trend; rich call premium"
    elif s >= 0.50:
        level, delta, dte, note = "Moderate", 0.25, "14–30 days", "Mixed; balance income vs upside"
    else:
        level, delta, dte, note = "Defensive", 0.15, "21–45 days", "Strong uptrend / limited premium"
    return {"score": round(s*100), "level": level, "delta": delta, "dte": dte, "note": note}
