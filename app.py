# app.py
from __future__ import annotations

import os
import streamlit as st
from loaders.fred_loader import load_fred_series, load_extra_series, FRED, API_KEY
from loaders.yf_loader import load_equity, load_vix

from dotenv import load_dotenv

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="U.S. Market Risk & Options Dashboard", page_icon="📊", layout="wide")
APP_TITLE = "📊 U.S. Market Risk & Options Dashboard"

# ── Config / constants ────────────────────────────────────────────────────────
# If you have a config.py, we’ll use it; otherwise fallback to local defaults.
try:
    from config import FRED_SERIES, START_EQ_DATE
except Exception:
    START_EQ_DATE = "2018-01-01"
    FRED_SERIES = {
        "M1": "M1SL",
        "M2": "M2SL",
        "10Y": "DGS10",
    }

# ── Secret loading (local .env first, then Streamlit Cloud secrets) ───────────
load_dotenv()  # local dev: reads .env if present

def get_secret(name: str, default: str = "") -> str:
    """Prefer environment/.env locally; fall back to Streamlit Cloud secrets."""
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets[name]  # works on Streamlit Cloud
    except Exception:
        return default

FRED_API_KEY = get_secret("FRED_API_KEY", "")

def _valid_fred_key(k: str) -> bool:
    return bool(k) and len(k) == 32 and k.islower() and k.isalnum()

# ── Imports that rely on the API key existing ─────────────────────────────────
from fredapi import Fred
from panels.csp_score_panel import render_csp_score_panel
from panels.csp_scanner_panel import render_csp_scanner
from panels.heatmap_panel import render_heatmap          # returns metrics dict
from panels.forecast_panel import render_forecast_panel  # expects metrics
from panels.options_panel import render_options_panel    # expects extra & spy_trend
from panels.timeseries_panel import render_base_charts, render_mmf_section

from loaders.fred_loader import load_fred_series, load_extra_series, FRED, API_KEY
from loaders.yf_loader import load_equity, load_vix
from compute.metrics import trend_slope_percent  # the same helper you use elsewhere



# ── Sidebar controls (before data loads) ──────────────────────────────────────
with st.sidebar:
    st.subheader("Settings")
    st.caption("Data is cached ~30–60 minutes depending on source.")

    # Secrets status (masked)
    st.write("FRED key:", (FRED_API_KEY[:4] + "…" + FRED_API_KEY[-4:]) if _valid_fred_key(FRED_API_KEY) else "❌ Missing")

    st.divider()
    auto_mmf = st.toggle("Auto-download ICI MMF flows", value=True)
    st.caption("If off, upload the file manually below.")
    mmf_file = st.file_uploader("ICI MMF flows file (CSV/XLS/XLSX)", type=["csv", "xls", "xlsx"])

# ── Title ─────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)

# ── Option Aggressiveness at the TOP ──────────────────────────────────────────
st.markdown("## ⚖️ Option Aggressiveness (CSP / Covered Calls)")

# Minimal inputs for this panel
spx_for_trend = load_equity("^GSPC", start=START_EQ_DATE)
spy_trend_top = trend_slope_percent(spx_for_trend)  # % slope based on 50–200d or your config
extra_top     = load_extra_series()                  # { "vix": Series, "vix3m": Series, "BaaSpread", etc. }

# Render the combined panel at the top (CSP left, CC right)
from panels.options_panel import render_options_panel
render_options_panel(extra_top, spy_trend=spy_trend_top)

st.markdown("---")
# ── Validate FRED key (show UI error but don’t crash earlier UI) ──────────────
if not _valid_fred_key(FRED_API_KEY):
    st.error(
        "FRED API key missing/invalid. "
        "Locally, set it in a `.env` file as `FRED_API_KEY=your32charlowercasekey`. "
        "On Streamlit Cloud, add it under **Settings → Secrets**."
    )
    st.stop()

# ── Initialize FRED client ────────────────────────────────────────────────────
fred = Fred(api_key=FRED_API_KEY)

# ── Load core time series ─────────────────────────────────────────────────────
try:
    m1 = load_fred_series(FRED_SERIES["M1"])
    m2 = load_fred_series(FRED_SERIES["M2"])
    y10 = load_fred_series(FRED_SERIES["10Y"])
except Exception as e:
    st.error(f"FRED data load failed: {e}")
    st.stop()

spx = load_equity("^GSPC", start=START_EQ_DATE)
vix = load_vix(start=START_EQ_DATE)

# ── Risk heat map & Macro forecast panels ─────────────────────────────────────
metrics = render_heatmap(m1, m2, y10, spx, vix)   # metrics includes spy_trend, vix, etc.
render_forecast_panel(metrics)

st.markdown("---")

render_csp_score_panel()

st.markdown("### 🔎 CSP Scanner")
render_csp_scanner()

st.markdown("---")



# # ── Options panel (uses extra macro inputs + trend) ───────────────────────────
# extra = load_extra_series()                       # dict with "2Y", "Claims", "LEI", "BaaSpread"
# render_options_panel(extra, spy_trend=metrics.get("spy_trend"))

# ── Base charts (M1/M2/10Y/SPX/VIX) ───────────────────────────────────────────
render_base_charts(m1, m2, y10, spx, vix)

# ── Money Market flows section (auto-download or manual upload) ───────────────
render_mmf_section(auto_mmf=auto_mmf, mmf_file=mmf_file)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Notes: Risk buckets are simplified heuristics; tune thresholds in `config.py`. "
    "Data sources: FRED (M1, M2, DGS10, DGS2, ICSA, USSLIND, BAA10YM), Yahoo Finance (^GSPC, ^VIX, DXY). "
    "Charts use `width='stretch'` to adapt to layout."
)
