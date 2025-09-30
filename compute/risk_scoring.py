"""Risk heat map buckets and composite macro risk score."""

from __future__ import annotations

import pandas as pd

from config import THRESHOLDS
from compute.metrics import pct_change_yoy, trend_slope_percent


def color_chip(level: str) -> str:
    chips = {"Green": "Green", "Yellow": "Yellow", "Red": "Red"}
    return chips.get(level, level)


def risk_bucket_high_bad(value, green_max, yellow_max):
    if value <= green_max:
        return "Green"
    if value <= yellow_max:
        return "Yellow"
    return "Red"


def risk_bucket_low_bad(value, yellow_min, green_min):
    if value >= green_min:
        return "Green"
    if value >= yellow_min:
        return "Yellow"
    return "Red"


def score_panel(m1: pd.Series, m2: pd.Series, y10: pd.Series, spx: pd.Series, vix: pd.Series):
    m1_yoy = pct_change_yoy(m1)
    m2_yoy = pct_change_yoy(m2)
    vix_last = vix.dropna().iloc[-1].item() if not vix.empty else float("nan")
    y10_last = y10.dropna().iloc[-1].item() if not y10.empty else float("nan")
    spy_trend = trend_slope_percent(spx)

    m1_bucket = risk_bucket_high_bad(m1_yoy, THRESHOLDS["M1_yoy"]["green_max"], THRESHOLDS["M1_yoy"]["yellow_max"]) if pd.notna(m1_yoy) else "Yellow"
    m2_bucket = risk_bucket_high_bad(m2_yoy, THRESHOLDS["M2_yoy"]["green_max"], THRESHOLDS["M2_yoy"]["yellow_max"]) if pd.notna(m2_yoy) else "Yellow"
    y10_bucket = risk_bucket_high_bad(y10_last, THRESHOLDS["10Y"]["green_max"], THRESHOLDS["10Y"]["yellow_max"]) if pd.notna(y10_last) else "Yellow"
    vix_bucket = risk_bucket_high_bad(vix_last, THRESHOLDS["VIX"]["green_max"], THRESHOLDS["VIX"]["yellow_max"]) if pd.notna(vix_last) else "Yellow"
    trend_bucket = risk_bucket_low_bad(spy_trend, THRESHOLDS["SPY_trend"]["yellow_min"], THRESHOLDS["SPY_trend"]["green_min"]) if pd.notna(spy_trend) else "Yellow"

    rows = [
        {"Factor": "10Y Yield", "Latest": f"{y10_last:.2f}%", "Signal": color_chip(y10_bucket), "Detail": "Higher yield = tougher valuations"},
        {"Factor": "M2 YoY", "Latest": f"{m2_yoy:.2f}%", "Signal": color_chip(m2_bucket), "Detail": "YoY money growth"},
        {"Factor": "M1 YoY", "Latest": f"{m1_yoy:.2f}%", "Signal": color_chip(m1_bucket), "Detail": "Transaction money growth"},
        {"Factor": "S&P Trend (50d-200d)", "Latest": f"{spy_trend:.2f}%", "Signal": color_chip(trend_bucket), "Detail": "Momentum proxy"},
        {"Factor": "VIX", "Latest": f"{vix_last:.2f}", "Signal": color_chip(vix_bucket), "Detail": "Higher VIX = stress"},
    ]
    return pd.DataFrame(rows), {"m1_yoy": m1_yoy, "m2_yoy": m2_yoy, "y10": y10_last, "vix": vix_last, "spy_trend": spy_trend}


def composite_risk_score(y10, y2, vix, claims, spread, dxy_level, spy_trend):
    """Rough normalization to 0-100 (higher = riskier)."""
    score = 0.0
    count = 0

    def add(value):
        nonlocal score, count
        if pd.notna(value):
            score += value
            count += 1

    add(min(100.0, (y10 - 3.5) * 40.0) if pd.notna(y10) else None)
    add(min(100.0, (y2 - 3.5) * 40.0) if pd.notna(y2) else None)
    add(min(100.0, (vix - 15.0) * 5.0) if pd.notna(vix) else None)
    add(min(100.0, (claims / 250000.0 - 1.0) * 100.0) if pd.notna(claims) else None)
    add(min(100.0, spread * 20.0) if pd.notna(spread) else None)
    add(min(100.0, (dxy_level / 100.0 - 1.0) * 50.0) if pd.notna(dxy_level) else None)
    add(max(0.0, -spy_trend * 20.0) if pd.notna(spy_trend) else None)

    return round(score / count, 1) if count else float("nan")
