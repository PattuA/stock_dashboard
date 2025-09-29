from dotenv import load_dotenv
import os
import streamlit as st

from pathlib import Path

from config import APP_TITLE, START_EQ_DATE, FRED_SERIES
from loaders.fred_loader import load_fred_series, load_extra_series, FRED, API_KEY
from loaders.yf_loader import load_equity, load_vix
from panels.heatmap_panel import render_heatmap
from panels.forecast_panel import render_forecast_panel
from panels.options_panel import render_options_panel
from panels.timeseries_panel import render_base_charts, render_mmf_section
from panels.csp_scanner_panel import render_csp_scanner
from panels.csp_score_panel import render_csp_score_panel

from dotenv import load_dotenv
load_dotenv()
def get_secret(name: str, default: str = "") -> str:
    """Prefer env var; if missing, try Streamlit secrets; otherwise default."""
    # 1) local dev: .env / environment
    val = os.getenv(name)
    if val:
        return val
    # 2) cloud: st.secrets (guarded to avoid parsing when file not present)
    try:
        return st.secrets[name]  # will work on Streamlit Cloud
    except Exception:
        return default

FRED_API_KEY = get_secret("FRED_API_KEY", "")
if not (len(FRED_API_KEY) == 32 and FRED_API_KEY.isalnum() and FRED_API_KEY.islower()):
    st.error(
        "FRED API key missing/invalid. Set it in a local `.env` (FRED_API_KEY=...) "
        "or in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()

st.cache_data.clear()
st.set_page_config(page_title="Market Risk Dashboard", page_icon="📊", layout="wide")
st.title(APP_TITLE)

# Key check for FRED API
if FRED is None:
    st.error("FRED API key missing/invalid. Create a `.env` next to app.py with:\n\nFRED_API_KEY=your32charlowercasekey")
    st.stop()

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    st.caption("Data refresh ~every 30 minutes (cached).")
    st.write("FRED key loaded (masked):", FRED_API_KEY[:4] + "..." + FRED_API_KEY[-4:])
    st.markdown("---")
    auto_mmf = st.toggle("Auto-download ICI MMF flows", value=True)
    st.caption("If off, upload the file below.")
    mmf_file = st.file_uploader("ICI MMF flows file (CSV/XLS)", type=["csv", "xls", "xlsx"])

# Load base series
try:
    m1 = load_fred_series(FRED_SERIES["M1"])
    m2 = load_fred_series(FRED_SERIES["M2"])
    y10 = load_fred_series(FRED_SERIES["10Y"])
except Exception as e:
    st.error(f"FRED data load failed: {e}")
    st.stop()

spx = load_equity("^GSPC", start=START_EQ_DATE)
vix = load_vix(start=START_EQ_DATE)

# Panels
metrics = render_heatmap(m1, m2, y10, spx, vix)
render_forecast_panel(metrics)

# options panel needs extra & spy_trend
extra = load_extra_series()
render_options_panel(extra, spy_trend=metrics["spy_trend"])

render_csp_scanner()

render_csp_score_panel()

render_base_charts(m1, m2, y10, spx, vix)
render_mmf_section(auto_mmf, mmf_file)

st.markdown("---")
st.caption(
    "Notes: Risk buckets are simplified heuristics. Tune thresholds in config.py. "
    "Data from FRED (M1, M2, DGS10, DGS2, ICSA, USSLIND, BAA10YM) and Yahoo Finance (^GSPC, ^VIX, DXY)."
)
