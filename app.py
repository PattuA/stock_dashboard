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
    st.write("FRED key loaded (masked):", API_KEY[:4] + "..." + API_KEY[-4:])
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


# ... inside app layout, after other panels:
render_csp_scanner()

render_base_charts(m1, m2, y10, spx, vix)
render_mmf_section(auto_mmf, mmf_file)

st.markdown("---")
st.caption(
    "Notes: Risk buckets are simplified heuristics. Tune thresholds in config.py. "
    "Data from FRED (M1, M2, DGS10, DGS2, ICSA, USSLIND, BAA10YM) and Yahoo Finance (^GSPC, ^VIX, DXY)."
)
