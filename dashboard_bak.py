# 📊 Stock Market Risk Dashboard (Streamlit)
# Windows-friendly, single-file app
# Requirements: streamlit, fredapi, yfinance, pandas, python-dotenv

import os, re, io
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import numpy as np  # <-- add near the top
import requests
from bs4 import BeautifulSoup

# ---------- Config ----------
APP_TITLE = "📊 U.S. Market Risk Dashboard"
START_EQ_DATE = "2018-01-01"  # history window for equities
FRED_SERIES = {
    "M1": "M1SL",
    "M2": "M2SL",
    "10Y": "DGS10",  # 10-Year Treasury Yield (%)
}
# Risk thresholds (simple defaults; tune as you like)
THRESHOLDS = {
    "10Y": {"green_max": 3.75, "yellow_max": 4.25},      # % yield
    "M2_yoy": {"green_max": 5.0, "yellow_max": 9.0},     # % YoY
    "M1_yoy": {"green_max": 5.0, "yellow_max": 9.0},     # % YoY
    "SPY_trend": {"green_min": 0.0, "yellow_min": -2.0}, # % 50d vs 200d slope proxy
    "VIX": {"green_max": 16.0, "yellow_max": 22.0},      # level
}

FRED_SERIES_EXTRA = {
    "2Y": "DGS2",          # 2-Year Treasury yield
    "Claims": "ICSA",      # Weekly unemployment claims
    "LEI": "USSLIND",      # Leading Economic Index (Conference Board via FRED)
    "BaaSpread": "BAA10YM" # Moody's Baa vs 10Y Treasury spread
}

ICI_MMF_PAGE = "https://www.ici.org/research/stats/mmf"

@st.cache_data(ttl=60*30, show_spinner=False)
def load_extra_series() -> dict[str, pd.Series]:
    out = {}
    for name, sid in FRED_SERIES_EXTRA.items():
        try:
            s = fred.get_series(sid).dropna()
            out[name] = s
        except Exception as e:
            st.warning(f"Could not load {name} ({sid}): {e}")
    return out

@st.cache_data(ttl=60*30, show_spinner=False)
def load_dxy(start=START_EQ_DATE) -> pd.Series:
    # Dollar Index from Yahoo; return a named Series
    df = yf.download("DX-Y.NYB", start=start, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float, name="DXY")
    return df["Close"].dropna().rename("DXY")

def composite_risk_score(y10, y2, vix, claims, spread, dxy, spy_trend):
    """
    Normalize each factor into 0–100 risk contribution.
    Higher score = higher risk.
    """
    score = 0
    factors = 0
    
    if pd.notna(y10):  # 10Y yield
        score += min(100, (y10 - 3.5) * 40)  # risk climbs above 3.5%
        factors += 1
    if pd.notna(y2):   # 2Y yield
        score += min(100, (y2 - 3.5) * 40)
        factors += 1
    if pd.notna(vix):  # VIX
        score += min(100, (vix - 15) * 5)
        factors += 1
    if pd.notna(claims):  # Unemployment claims
        score += min(100, (claims / 250000 - 1) * 100)
        factors += 1
    if pd.notna(spread):  # Credit spreads
        score += min(100, spread * 20)
        factors += 1
    if pd.notna(dxy):  # Dollar Index
        score += min(100, (dxy / 100 - 1) * 50)
        factors += 1
    if pd.notna(spy_trend):  # Trend slope proxy
        score += max(0, -spy_trend * 20)  # negative momentum = risk
        factors += 1
    return round(score / factors, 1) if factors > 0 else float("nan")

@st.cache_data(ttl=60*60*24, show_spinner=True)  # cache for 1 day
def fetch_latest_ici_xls_url() -> str | None:
    """
    Scrape ICI MMF stats page and return the first .xls link (e.g., mm_summary_data_2025.xls).
    """
    resp = requests.get(ICI_MMF_PAGE, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    # find links ending with .xls (ICI usually posts an xls named like 'mm_summary_data_YYYY.xls')
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".xls"):
            # Make absolute if needed
            if href.startswith("http"):
                return href
            # handle relative paths
            from urllib.parse import urljoin
            return urljoin(ICI_MMF_PAGE, href)
    return None

@st.cache_data(ttl=60*60*24, show_spinner=True)
def load_latest_ici_mmf_flows() -> pd.DataFrame:
    """
    Download the latest ICI MMF .xls and return a tidy DataFrame with a datetime index.
    Tries to auto-detect a date column and a numeric column (flows or assets).
    """
    url = fetch_latest_ici_xls_url()
    if not url:
        raise RuntimeError("Could not find an .xls link on ICI page.")
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    # Read Excel from bytes using xlrd engine
    with io.BytesIO(r.content) as bio:
        # some files have multiple sheets; read all and pick the first usable one
        xls = pd.ExcelFile(bio, engine="xlrd")
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            if df.empty or df.shape[1] < 2:
                continue
            # normalize headers
            df.columns = [str(c).strip() for c in df.columns]
            # find date col
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            if date_col is None:
                # sometimes first col is a date without header; try first column
                date_col = df.columns[0]
            # coerce dates
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            # choose a numeric column (prefer flows/net/assets)
            numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                continue
            # heuristic preference order
            def score(c):
                lc = c.lower()
                return (
                    ("flow" in lc or "net" in lc) * 2
                    + ("asset" in lc or "total" in lc) * 1
                )
            numeric_cols.sort(key=score, reverse=True)
            out = df[[date_col] + numeric_cols].copy()
            out = out.set_index(date_col)
            out.index.name = "Date"
            return out
    raise RuntimeError("Could not parse any sheet in the ICI workbook.")

# ---------- Helpers ----------
def color_chip(level: str) -> str:
    return {"Green": "🟢 Green", "Yellow": "🟡 Yellow", "Red": "🔴 Red"}[level]

def bucket_from_value(value, kind="high_bad"):
    """
    kind = 'high_bad' → higher is worse (e.g., 10Y yield, VIX, YoY too hot)
    kind = 'low_bad'  → lower is worse (e.g., trend momentum)
    Uses THRESHOLDS keys present in caller.
    """
    # This is just a utility; actual bucketing happens in score_* functions below
    return value

def risk_bucket_high_bad(value, green_max, yellow_max):
    if value <= green_max:
        return "Green"
    elif value <= yellow_max:
        return "Yellow"
    return "Red"

def risk_bucket_low_bad(value, yellow_min, green_min):
    # Here, higher is better (e.g., positive trend). If it drops below yellow_min → Red
    if value >= green_min:
        return "Green"
    elif value >= yellow_min:
        return "Yellow"
    return "Red"

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_fred_series(series_id: str) -> pd.Series:
    """Fetch a series from FRED as a pd.Series (datetime index)."""
    s = fred.get_series(series_id)  # returns Series with DatetimeIndex (monthly/daily)
    s = s.dropna()
    return s

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_equity(symbol="^GSPC", start=START_EQ_DATE) -> pd.Series:
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].dropna()

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_vix(start=START_EQ_DATE) -> pd.Series:
    df = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].dropna()

def pct_change_yoy(s: pd.Series) -> float:
    """Compute year-over-year % change from last value vs value 12 months ago."""
    if s.empty:
        return float("nan")
    s_m = s.resample("ME").last()
    if len(s_m) < 13:
        return float("nan")
    last = s_m.iloc[-1]
    prev = s_m.iloc[-13]
    return (last / prev - 1.0) * 100.0


def trend_slope_percent(series: pd.Series, window_long=200, window_short=50) -> float:
    """Rough trend proxy: (short_MA - long_MA) / long_MA * 100 (last value)."""
    # Enough history?
    need = max(window_long, window_short)
    if series is None or len(series) < need:
        return float("nan")

    # Force scalar ends using min_periods and dropna
    sma_long = series.rolling(window_long, min_periods=window_long).mean()
    sma_short = series.rolling(window_short, min_periods=window_short).mean()

    if sma_long.dropna().empty or sma_short.dropna().empty:
        return float("nan")

    # Cast to float to avoid Series ambiguity
    last_long  = sma_long.dropna().iloc[-1].item()
    last_short = sma_short.dropna().iloc[-1].item()
    if not np.isfinite(last_long) or last_long == 0.0:
        return float("nan")

    return (last_short - last_long) / last_long * 100.0


def score_panel(m1, m2, y10, spx, vix):
    # Levels/growth
    m1_yoy = pct_change_yoy(m1)
    m2_yoy = pct_change_yoy(m2)
    vix_last   = vix.dropna().iloc[-1].item()  if not vix.empty else float("nan")
    y10_last   = y10.dropna().iloc[-1].item()  if not y10.empty else float("nan")
    spy_trend = trend_slope_percent(spx)

    # Buckets
    m1_bucket = risk_bucket_high_bad(m1_yoy, THRESHOLDS["M1_yoy"]["green_max"], THRESHOLDS["M1_yoy"]["yellow_max"]) if pd.notna(m1_yoy) else "Yellow"
    m2_bucket = risk_bucket_high_bad(m2_yoy, THRESHOLDS["M2_yoy"]["green_max"], THRESHOLDS["M2_yoy"]["yellow_max"]) if pd.notna(m2_yoy) else "Yellow"
    y10_bucket = risk_bucket_high_bad(y10_last, THRESHOLDS["10Y"]["green_max"], THRESHOLDS["10Y"]["yellow_max"]) if pd.notna(y10_last) else "Yellow"
    vix_bucket = risk_bucket_high_bad(vix_last, THRESHOLDS["VIX"]["green_max"], THRESHOLDS["VIX"]["yellow_max"]) if pd.notna(vix_last) else "Yellow"
    trend_bucket = risk_bucket_low_bad(spy_trend, THRESHOLDS["SPY_trend"]["yellow_min"], THRESHOLDS["SPY_trend"]["green_min"]) if pd.notna(spy_trend) else "Yellow"

    rows = [
        {"Factor": "10Y Yield", "Latest": f"{y10_last:.2f}%", "Signal": color_chip(y10_bucket), "Detail": "Higher yield = tougher valuations"},
        {"Factor": "M2 YoY", "Latest": f"{m2_yoy:.2f}%", "Signal": color_chip(m2_bucket), "Detail": "Too-hot YoY can signal inflation/liquidity shifts"},
        {"Factor": "M1 YoY", "Latest": f"{m1_yoy:.2f}%", "Signal": color_chip(m1_bucket), "Detail": "Transaction money growth"},
        {"Factor": "S&P Trend (50d-200d)", "Latest": f"{spy_trend:.2f}%", "Signal": color_chip(trend_bucket), "Detail": "Momentum proxy"},
        {"Factor": "VIX", "Latest": f"{vix_last:.2f}", "Signal": color_chip(vix_bucket), "Detail": "Higher VIX = stress"},
    ]
    return pd.DataFrame(rows), {
        "m1_yoy": m1_yoy, "m2_yoy": m2_yoy, "y10": y10_last, "vix": vix_last, "spy_trend": spy_trend
    }

def composite_risk_score(y10, y2, vix, claims, spread, dxy, spy_trend):
    """
    Normalize each factor into 0–100 risk contribution.
    Higher score = higher risk.
    """
    score = 0
    factors = 0
    
    if pd.notna(y10):  # 10Y yield
        score += min(100, (y10 - 3.5) * 40)  # risk climbs above 3.5%
        factors += 1
    if pd.notna(y2):   # 2Y yield
        score += min(100, (y2 - 3.5) * 40)
        factors += 1
    if pd.notna(vix):  # VIX
        score += min(100, (vix - 15) * 5)
        factors += 1
    if pd.notna(claims):  # Unemployment claims
        score += min(100, (claims / 250000 - 1) * 100)
        factors += 1
    if pd.notna(spread):  # Credit spreads
        score += min(100, spread * 20)
        factors += 1
    if pd.notna(dxy):  # Dollar Index
        score += min(100, (dxy / 100 - 1) * 50)
        factors += 1
    if pd.notna(spy_trend):  # Trend slope proxy
        score += max(0, -spy_trend * 20)  # negative momentum = risk
        factors += 1

    return round(score / factors, 1) if factors > 0 else float("nan")


# ---------- App Start ----------
st.set_page_config(page_title="Market Risk Dashboard", page_icon="📊", layout="wide")
st.title(APP_TITLE)

# Load .env adjacent to this file (robust for cmd/VS Code/Streamlit)
env_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("FRED_API_KEY", "")

# Validate API key format
if not re.fullmatch(r"[a-z0-9]{32}", API_KEY or ""):
    st.error("FRED API key missing/invalid. Add it to a `.env` file next to this script as:\n\n`FRED_API_KEY=your32charlowercasekey`")
    st.stop()

# Init FRED
fred = Fred(api_key=API_KEY)

# Sidebar info
with st.sidebar:
    st.subheader("Settings")
    st.caption("Data refresh ~every 30 minutes (cached).")
    st.write("FRED key loaded (masked):", API_KEY[:4] + "..." + API_KEY[-4:])
    st.markdown("---")
    auto_mmf = st.toggle("Auto-download ICI MMF flows", value=True)
    st.caption("If off, you can upload the file manually below.")
    mmf_file = st.file_uploader("ICI MMF flows file (CSV/XLS)", type=["csv", "xls", "xlsx"])


# Load time series
try:
    m1 = load_fred_series(FRED_SERIES["M1"])
    m2 = load_fred_series(FRED_SERIES["M2"])
    y10 = load_fred_series(FRED_SERIES["10Y"])
except Exception as e:
    st.error(f"FRED data load failed: {e}")
    st.stop()

spx = load_equity("^GSPC", start=START_EQ_DATE)
vix = load_vix(start=START_EQ_DATE)

# Score & Heat Map
heat_df, metrics = score_panel(m1, m2, y10, spx, vix)

st.subheader("Risk Heat Map")
st.dataframe(heat_df, width='stretch', hide_index=True)

# Summary metrics row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("10Y", f"{metrics['y10']:.2f}%")
c2.metric("M2 YoY", f"{metrics['m2_yoy']:.2f}%")
c3.metric("M1 YoY", f"{metrics['m1_yoy']:.2f}%")
c4.metric("S&P Trend 50–200", f"{metrics['spy_trend']:.2f}%")
c5.metric("VIX", f"{metrics['vix']:.2f}")

st.markdown("---")

st.markdown("---")
st.subheader("🔮 Forecast Panel (Macro + Market Mix)")

extra = load_extra_series()
dxy = load_dxy()

# Extract latest values
y10_last = metrics["y10"]
spy_trend = metrics["spy_trend"]
y2_last = extra.get("2Y", pd.Series()).dropna().iloc[-1].item() if "2Y" in extra and not extra["2Y"].empty else float("nan")
claims_last = extra.get("Claims", pd.Series()).dropna().iloc[-1].item() if "Claims" in extra and not extra["Claims"].empty else float("nan")
lei_last = extra.get("LEI", pd.Series()).dropna().iloc[-1].item() if "LEI" in extra and not extra["LEI"].empty else float("nan")
spread_last = extra.get("BaaSpread", pd.Series()).dropna().iloc[-1].item() if "BaaSpread" in extra and not extra["BaaSpread"].empty else float("nan")
dxy_last = dxy.iloc[-1].item() if not dxy.empty else float("nan")

risk_score = composite_risk_score(y10_last, y2_last, metrics["vix"], claims_last, spread_last, dxy_last, spy_trend)

c1, c2, c3 = st.columns(3)
c1.metric("2Y Yield", f"{y2_last:.2f}%" if pd.notna(y2_last) else "N/A")
c2.metric("Unemployment Claims", f"{claims_last:,.0f}" if pd.notna(claims_last) else "N/A")
c3.metric("Baa Spread", f"{spread_last:.2f}" if pd.notna(spread_last) else "N/A")

c4, c5, c6 = st.columns(3)
c4.metric("LEI (Index)", f"{lei_last:.1f}" if pd.notna(lei_last) else "N/A")
c5.metric("Dollar Index (DXY)", f"{dxy_last:.1f}" if pd.notna(dxy_last) else "N/A")
c6.metric("Composite Risk Score", f"{risk_score:.1f}/100" if pd.notna(risk_score) else "N/A")

# Chart some key extra series
st.markdown("**Macro Indicators (Recent)**")
colX, colY = st.columns(2)
# Left column
with colX:
    st.markdown("**2Y Treasury Yield**")
    if "2Y" in extra and not extra["2Y"].empty:
        st.line_chart(extra["2Y"])
    else:
        st.write("No 2Y data")

    st.markdown("**Unemployment Claims**")
    if "Claims" in extra and not extra["Claims"].empty:
        st.line_chart(extra["Claims"].rename("Weekly Claims"))
    else:
        st.write("No Claims data")

# Right column
with colY:
    st.markdown("**Baa – 10Y Treasury Spread**")
    if "BaaSpread" in extra and not extra["BaaSpread"].empty:
        st.line_chart(extra["BaaSpread"].rename("Baa-10Y Spread"))
    else:
        st.write("No Spread data")

    st.markdown("**US Dollar Index (DXY)**")
    if dxy is not None and not dxy.empty:
        st.line_chart(dxy.rename("DXY"))
    else:
        st.write("No DXY data")


# Charts
st.subheader("Time Series")
colA, colB = st.columns(2)
with colA:
    st.markdown("**M1 & M2 (Monthly, last value)**")
    # Convert to monthly last for smoother viewing
    m1m = m1.resample("ME").last()
    m2m = m2.resample("ME").last()
    df_m = pd.DataFrame({"M1": m1m, "M2": m2m})
    st.line_chart(df_m)

with colB:
    st.markdown("**10-Year Treasury Yield (%)**")
    st.line_chart(y10)

colC, colD = st.columns(2)
with colC:
    st.markdown("**S&P 500 (Close)**")
    st.line_chart(spx)

with colD:
    st.markdown("**VIX (Close)**")
    st.line_chart(vix)

# Optional: Money Market flows upload
st.markdown("---")
st.subheader("Money Market Fund (MMF) Flows")

mmf_df = None
mmf_msg = ""
try:
    if auto_mmf:
        mmf_df = load_latest_ici_mmf_flows()
        mmf_msg = "Auto-downloaded latest ICI MMF workbook."
    elif mmf_file is not None:
        if mmf_file.name.lower().endswith(".csv"):
            mmf_df = pd.read_csv(mmf_file)
        else:
            mmf_df = pd.read_excel(mmf_file)  # pandas will pick engine
        mmf_msg = f"Loaded from upload: {mmf_file.name}"
except Exception as e:
    st.warning(f"MMF auto-download/parse failed: {e}. You can upload the file manually above.")

if mmf_df is not None:
    # Try to normalize again in case of manual upload
    if "Date" not in mmf_df.columns:
        # run a lightweight normalizer similar to loader
        mmf_df.columns = [str(c).strip() for c in mmf_df.columns]
        date_col = next((c for c in mmf_df.columns if "date" in c.lower()), mmf_df.columns[0])
        mmf_df[date_col] = pd.to_datetime(mmf_df[date_col], errors="coerce")
        mmf_df = mmf_df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        mmf_df.index.name = "Date"
    st.caption(mmf_msg)

    # Pick top 1–2 numeric columns to plot
    num_cols = [c for c in mmf_df.columns if pd.api.types.is_numeric_dtype(mmf_df[c])]
    if not num_cols:
        st.warning("No numeric columns found to chart; showing table.")
        st.dataframe(mmf_df.tail(50), width="stretch")
    else:
        st.write("Detected numeric columns:", ", ".join(num_cols[:3]))
        st.line_chart(mmf_df[num_cols[:2]].tail(260))  # ~5 years weekly
else:
    st.info("Enable **Auto-download ICI MMF flows** or upload the file to see the chart.")

# Footer
st.markdown("---")
st.caption(
    "Notes: Risk buckets are simplified heuristics. "
    "Tune thresholds in THRESHOLDS for your strategy. "
    "Data from FRED (M1, M2, DGS10) and Yahoo Finance (^GSPC, ^VIX)."
)
