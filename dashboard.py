# 📊 U.S. Market Risk Dashboard — Streamlit
# Windows-friendly, single-file app

import io
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fredapi import Fred

# =========================
# Config
# =========================
APP_TITLE = "📊 U.S. Market Risk Dashboard"
START_EQ_DATE = "2018-01-01"  # history window for equities

FRED_SERIES = {
    "M1": "M1SL",
    "M2": "M2SL",
    "10Y": "DGS10",  # 10-Year Treasury Yield (%)
}

# Risk thresholds (tune to taste)
THRESHOLDS = {
    "10Y": {"green_max": 3.75, "yellow_max": 4.25},       # % yield
    "M2_yoy": {"green_max": 5.0, "yellow_max": 9.0},      # % YoY
    "M1_yoy": {"green_max": 5.0, "yellow_max": 9.0},      # % YoY
    "SPY_trend": {"green_min": 0.0, "yellow_min": -2.0},  # % 50d vs 200d slope proxy
    "VIX": {"green_max": 16.0, "yellow_max": 22.0},       # level
}

# Extra macro series (FRED IDs)
FRED_SERIES_EXTRA = {
    "2Y": "DGS2",            # 2-Year Treasury yield
    "Claims": "ICSA",        # Initial unemployment claims (weekly)
    "LEI": "USSLIND",        # Leading Economic Index
    "BaaSpread": "BAA10YM",  # Moody's Baa - 10Y Treasury spread
}

ICI_MMF_PAGE = "https://www.ici.org/research/stats/mmf"

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Market Risk Dashboard", page_icon="📊", layout="wide")
st.title(APP_TITLE)

# =========================
# Env & FRED init
# =========================
env_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("FRED_API_KEY", "")
if not re.fullmatch(r"[a-z0-9]{32}", API_KEY or ""):
    st.error(
        "FRED API key missing/invalid. Create `.env` (next to this file) with:\n\n"
        "FRED_API_KEY=f8ddc7320babacdc6efce526d99ecb28"
    )
    st.stop()

fred = Fred(api_key=API_KEY)

# =========================
# Helpers
# =========================
def color_chip(level: str) -> str:
    return {"Green": "🟢 Green", "Yellow": "🟡 Yellow", "Red": "🔴 Red"}[level]

def risk_bucket_high_bad(value, green_max, yellow_max):
    if value <= green_max:
        return "Green"
    elif value <= yellow_max:
        return "Yellow"
    return "Red"

def risk_bucket_low_bad(value, yellow_min, green_min):
    # Higher is better; very low is bad
    if value >= green_min:
        return "Green"
    elif value >= yellow_min:
        return "Yellow"
    return "Red"

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_fred_series(series_id: str) -> pd.Series:
    s = fred.get_series(series_id)
    return s.dropna()

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_equity(symbol="^GSPC", start=START_EQ_DATE) -> pd.Series:
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    return df["Close"].dropna() if not df.empty else pd.Series(dtype=float)

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_vix(start=START_EQ_DATE) -> pd.Series:
    df = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    return df["Close"].dropna() if not df.empty else pd.Series(dtype=float)

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_dxy(start=START_EQ_DATE) -> pd.Series:
    # ICE Dollar Index continuous
    df = yf.download("DX-Y.NYB", start=start, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float, name="DXY")

    close = df.get("Close")
    if close is None:
        return pd.Series(dtype=float, name="DXY")

    # yfinance can return a Series or a DataFrame (multi-column) for Close
    if isinstance(close, pd.DataFrame):
        # pick the first column with data
        s = close.loc[:, close.columns[0]]
    else:
        s = close

    s = s.dropna()
    s.name = "DXY"  # set name directly, no rename()
    return s

def pct_change_yoy(s: pd.Series) -> float:
    """YoY % change from last value vs value 12 months ago."""
    if s.empty:
        return float("nan")
    s_m = s.resample("ME").last()  # 'ME' = month-end (replaces deprecated 'M')
    if len(s_m) < 13:
        return float("nan")
    last = s_m.iloc[-1]
    prev = s_m.iloc[-13]
    return (last / prev - 1.0) * 100.0

def trend_slope_percent(series: pd.Series, window_long=200, window_short=50) -> float:
    """(short_MA - long_MA) / long_MA * 100 (last value)."""
    need = max(window_long, window_short)
    if series is None or len(series) < need:
        return float("nan")

    sma_long = series.rolling(window_long, min_periods=window_long).mean()
    sma_short = series.rolling(window_short, min_periods=window_short).mean()

    if sma_long.dropna().empty or sma_short.dropna().empty:
        return float("nan")

    last_long = sma_long.dropna().iloc[-1].item()
    last_short = sma_short.dropna().iloc[-1].item()

    if not np.isfinite(last_long) or last_long == 0.0:
        return float("nan")

    return (last_short - last_long) / last_long * 100.0

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
    return pd.DataFrame(rows), {
        "m1_yoy": m1_yoy, "m2_yoy": m2_yoy, "y10": y10_last, "vix": vix_last, "spy_trend": spy_trend
    }

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_extra_series() -> dict:
    out = {}
    for name, sid in FRED_SERIES_EXTRA.items():
        try:
            s = fred.get_series(sid).dropna()
            out[name] = s
        except Exception as e:
            st.warning(f"Could not load {name} ({sid}): {e}")
    return out

def composite_risk_score(y10, y2, vix, claims, spread, dxy_level, spy_trend):
    """
    Rough normalization to 0–100. Higher = riskier.
    """
    score = 0.0
    factors = 0

    if pd.notna(y10):
        score += min(100.0, (y10 - 3.5) * 40.0)
        factors += 1
    if pd.notna(y2):
        score += min(100.0, (y2 - 3.5) * 40.0)
        factors += 1
    if pd.notna(vix):
        score += min(100.0, (vix - 15.0) * 5.0)
        factors += 1
    if pd.notna(claims):
        score += min(100.0, (claims / 250000.0 - 1.0) * 100.0)
        factors += 1
    if pd.notna(spread):
        score += min(100.0, spread * 20.0)
        factors += 1
    if pd.notna(dxy_level):
        score += min(100.0, (dxy_level / 100.0 - 1.0) * 50.0)
        factors += 1
    if pd.notna(spy_trend):
        score += max(0.0, -spy_trend * 20.0)  # negative momentum increases risk
        factors += 1

    return round(score / factors, 1) if factors > 0 else float("nan")

# -------- ICI MMF auto-download (with manual upload fallback) --------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=True)
def fetch_latest_ici_xls_url() -> str:
    resp = requests.get(ICI_MMF_PAGE, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".xls"):
            if href.startswith("http"):
                return href
            from urllib.parse import urljoin
            return urljoin(ICI_MMF_PAGE, href)
    return ""

@st.cache_data(ttl=60 * 60 * 24, show_spinner=True)
def load_latest_ici_mmf_flows() -> pd.DataFrame:
    url = fetch_latest_ici_xls_url()
    if not url:
        raise RuntimeError("Could not find an .xls link on ICI page.")
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with io.BytesIO(r.content) as bio:
        xls = pd.ExcelFile(bio, engine="xlrd")
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            if df.empty or df.shape[1] < 2:
                continue
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                continue
            def score_col(c):
                lc = c.lower()
                return (("flow" in lc or "net" in lc) * 2) + (("asset" in lc or "total" in lc) * 1)
            numeric_cols.sort(key=score_col, reverse=True)
            out = df[[date_col] + numeric_cols].copy().set_index(date_col)
            out.index.name = "Date"
            return out
    raise RuntimeError("Could not parse any sheet in the ICI workbook.")

# ==== Extra market loaders (place near other @st.cache_data loaders) ====
@st.cache_data(ttl=60*30, show_spinner=False)
def load_yf_close(symbol: str, start=START_EQ_DATE, auto_adjust=True) -> pd.Series:
    df = yf.download(symbol, start=start, progress=False, auto_adjust=auto_adjust)
    if df.empty:
        return pd.Series(dtype=float)
    close = df.get("Close")
    if isinstance(close, pd.DataFrame):
        s = close.iloc[:, 0]
    else:
        s = close
    return s.dropna()


@st.cache_data(ttl=60*30, show_spinner=False)
def load_breadth_series(start=START_EQ_DATE) -> pd.Series:
    """RSP/SPY ratio as a breadth proxy."""
    rsp = load_yf_close("RSP", start=start, auto_adjust=True)
    spy = load_yf_close("SPY", start=start, auto_adjust=True)
    if rsp.empty or spy.empty:
        return pd.Series(dtype=float)
    ratio = (rsp / spy).dropna()
    ratio.name = "RSP/SPY"
    return ratio

@st.cache_data(ttl=60*30, show_spinner=False)
def load_vix_term(start=START_EQ_DATE) -> dict:
    """Return dict with Series for ^VIX (1M), ^VIX3M (3M), ^VIX9D (9-day).
    Forces Series output and sets a clean name for each.
    """
    def as_named_series(sym: str, name: str) -> pd.Series:
        df = yf.download(sym, start=start, progress=False, auto_adjust=False)
        if df.empty:
            return pd.Series(dtype=float, name=name)
        close = df.get("Close")
        if isinstance(close, pd.DataFrame):
            s = close.iloc[:, 0]
        else:
            s = close
        s = s.dropna()
        s.name = name
        return s

    vix   = as_named_series("^VIX",   "VIX")
    vix3m = as_named_series("^VIX3M", "VIX3M")
    vix9d = as_named_series("^VIX9D", "VIX9D")
    return {"vix": vix, "vix3m": vix3m, "vix9d": vix9d}



def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def zshape_for_premium(vix):
    """
    Map VIX to a 0..1 'premium attractiveness' score.
    Peaks roughly around 18-24, fades if too low (no premium) or too high (crash risk).
    """
    if pd.isna(vix): return 0.5
    if vix < 12:  return 0.1
    if vix < 16:  return 0.4
    if vix < 20:  return 0.8
    if vix < 26:  return 1.0
    if vix < 32:  return 0.6
    if vix < 40:  return 0.3
    return 0.1

def contango_score(vix, vix3m):
    """
    VIX term structure score: positive if VIX3M > VIX (contango), negative if inverted (backwardation).
    Returns 0..1
    """
    if pd.isna(vix) or pd.isna(vix3m): return 0.5
    spread = (vix3m - vix)
    if spread <= -2: return 0.0
    if spread <= 0:  return 0.3
    if spread <= 2:  return 0.7
    return 1.0

def credit_score(baa_spread):
    """Lower Baa-10Y spread is good (0..1)."""
    if pd.isna(baa_spread): return 0.5
    if baa_spread < 1.8: return 1.0
    if baa_spread < 2.3: return 0.8
    if baa_spread < 3.0: return 0.5
    if baa_spread < 4.0: return 0.2
    return 0.0

def trend_score(trend_pct):
    """Your existing SPY 50-200 slope; positive is good. Map to 0..1."""
    if pd.isna(trend_pct): return 0.5
    if trend_pct >= 2.0: return 1.0
    if trend_pct >= 0.5: return 0.8
    if trend_pct >= 0.0: return 0.6
    if trend_pct >= -1.0: return 0.3
    return 0.1

def breadth_score(rsp_spy_trend):
    """Breadth trend via RSP/SPY slope; positive = broad leadership (0..1)."""
    if pd.isna(rsp_spy_trend): return 0.5
    if rsp_spy_trend >= 0.5: return 1.0
    if rsp_spy_trend >= 0.0: return 0.7
    if rsp_spy_trend >= -0.5: return 0.4
    return 0.2

def slope_of_ratio(series: pd.Series, short=50, long=200) -> float:
    """Percent slope using your existing method on a ratio series."""
    if series is None or len(series) < max(short, long): return float("nan")
    sma_s = series.rolling(short, min_periods=short).mean()
    sma_l = series.rolling(long,  min_periods=long).mean()
    if sma_l.dropna().empty or sma_s.dropna().empty: return float("nan")
    last_l = sma_l.dropna().iloc[-1]
    last_s = sma_s.dropna().iloc[-1]
    if pd.isna(last_l) or last_l == 0: return float("nan")
    return (last_s - last_l) / last_l * 100.0

def csp_guidance(vix, vix3m, spy_trend, breadth_trend, baa_spread):
    """
    Cash-Secured Puts guidance: returns dict with level, delta, DTE, notes.
    Aggressive when: VIX premium good, term in contango, trend/breadth not negative, credit tight.
    """
    s_prem   = zshape_for_premium(vix)
    s_term   = contango_score(vix, vix3m)
    s_trend  = trend_score(spy_trend)
    s_bread  = breadth_score(breadth_trend)
    s_credit = credit_score(baa_spread)

    score = (0.30*s_prem + 0.20*s_term + 0.25*s_trend + 0.15*s_bread + 0.10*s_credit)  # 0..1
    if score >= 0.75:
        level, delta, dte, note = "Aggressive", 0.30, "7–14 days", "Good premium, stable structure"
    elif score >= 0.50:
        level, delta, dte, note = "Moderate", 0.20, "14–30 days", "Mixed signals"
    else:
        level, delta, dte, note = "Defensive", 0.10, "30–45 days", "Elevated risk / weak structure"
    return {"score": round(score*100, 0), "level": level, "delta": delta, "dte": dte, "note": note}

def cc_guidance(vix, vix3m, spy_trend, breadth_trend, baa_spread):
    """
    Covered Calls guidance: more aggressive (nearer strikes/shorter DTE) when
    trend is flat/down, breadth weak, VIX elevated, or term inverted.
    """
    # Build a 'write-now' signal (higher = better time to write calls)
    s_vol    = zshape_for_premium(max(vix, 12))  # more vol = more premium
    s_term   = 1 - contango_score(vix, vix3m)    # backwardation favors capping upside
    s_trend  = 1 - trend_score(spy_trend)        # down/flat trend favors writing calls
    s_bread  = 1 - breadth_score(breadth_trend)  # weak breadth favors writing
    s_credit = 1 - credit_score(baa_spread)      # stress favors writing (if you're long stock)

    score = clamp(0.30*s_vol + 0.20*s_term + 0.30*s_trend + 0.10*s_bread + 0.10*s_credit, 0, 1)
    if score >= 0.75:
        level, delta, dte, note = "Aggressive", 0.35, "7–14 days", "Choppy/weak trend; rich call premium"
    elif score >= 0.50:
        level, delta, dte, note = "Moderate", 0.25, "14–30 days", "Mixed; balance income vs upside"
    else:
        level, delta, dte, note = "Defensive", 0.15, "21–45 days", "Strong uptrend / limited premium"
    return {"score": round(score*100, 0), "level": level, "delta": delta, "dte": dte, "note": note}


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("Settings")
    st.caption("Data refresh ~every 30 minutes (cached).")
    st.write("FRED key loaded (masked):", API_KEY[:4] + "..." + API_KEY[-4:])
    st.markdown("---")
    auto_mmf = st.toggle("Auto-download ICI MMF flows", value=True)
    st.caption("If off, upload the file below.")
    mmf_file = st.file_uploader("ICI MMF flows file (CSV/XLS)", type=["csv", "xls", "xlsx"])

# =========================
# Load core series
# =========================
try:
    m1 = load_fred_series(FRED_SERIES["M1"])
    m2 = load_fred_series(FRED_SERIES["M2"])
    y10 = load_fred_series(FRED_SERIES["10Y"])
except Exception as e:
    st.error(f"FRED data load failed: {e}")
    st.stop()

spx = load_equity("^GSPC", start=START_EQ_DATE)
vix = load_vix(start=START_EQ_DATE)

# =========================
# Heat Map
# =========================
heat_df, metrics = score_panel(m1, m2, y10, spx, vix)

st.subheader("Risk Heat Map")
st.dataframe(heat_df, width="stretch", hide_index=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("10Y", f"{metrics['y10']:.2f}%")
c2.metric("M2 YoY", f"{metrics['m2_yoy']:.2f}%")
c3.metric("M1 YoY", f"{metrics['m1_yoy']:.2f}%")
c4.metric("S&P Trend 50–200", f"{metrics['spy_trend']:.2f}%")
c5.metric("VIX", f"{metrics['vix']:.2f}")

st.markdown("---")

# =========================
# Forecast Panel
# =========================
st.subheader("🔮 Forecast Panel (Macro + Market Mix)")

extra = load_extra_series()
dxy = load_dxy()
dxy_last = dxy.iloc[-1].item() if not dxy.empty else float("nan")
if not dxy.empty:
    st.line_chart(dxy)  # already a named Series
else:
    st.write("No DXY data")

y10_last = metrics["y10"]
spy_trend = metrics["spy_trend"]
y2_last = extra["2Y"].dropna().iloc[-1].item() if "2Y" in extra and not extra["2Y"].empty else float("nan")
claims_last = extra["Claims"].dropna().iloc[-1].item() if "Claims" in extra and not extra["Claims"].empty else float("nan")
lei_last = extra["LEI"].dropna().iloc[-1].item() if "LEI" in extra and not extra["LEI"].empty else float("nan")
spread_last = extra["BaaSpread"].dropna().iloc[-1].item() if "BaaSpread" in extra and not extra["BaaSpread"].empty else float("nan")


risk_score = composite_risk_score(y10_last, y2_last, metrics["vix"], claims_last, spread_last, dxy_last, spy_trend)

# was: m1, m2, m3 = st.columns(3)
fc1, fc2, fc3 = st.columns(3)
fc1.metric("2Y Yield", f"{y2_last:.2f}%" if pd.notna(y2_last) else "N/A")
fc2.metric("Unemployment Claims", f"{claims_last:,.0f}" if pd.notna(claims_last) else "N/A")
fc3.metric("Baa Spread", f"{spread_last:.2f}" if pd.notna(spread_last) else "N/A")

# was: m4, m5, m6 = st.columns(3)
fc4, fc5, fc6 = st.columns(3)
fc4.metric("LEI (Index)", f"{lei_last:.1f}" if pd.notna(lei_last) else "N/A")
fc5.metric("Dollar Index (DXY)", f"{dxy_last:.1f}" if pd.notna(dxy_last) else "N/A")
fc6.metric("Composite Risk Score", f"{risk_score:.1f}/100" if pd.notna(risk_score) else "N/A")


st.markdown("**Macro Indicators (Recent)**")
colX, colY = st.columns(2)

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

with colY:
    st.markdown("**Baa – 10Y Treasury Spread**")
    if "BaaSpread" in extra and not extra["BaaSpread"].empty:
        st.line_chart(extra["BaaSpread"].rename("Baa-10Y Spread"))
    else:
        st.write("No Spread data")

    st.markdown("**US Dollar Index (DXY)**")
    if not dxy.empty:
        st.line_chart(dxy)  # already named "DXY"
    else:
        st.write("No DXY data")

st.markdown("---")

st.markdown("---")
st.subheader("🧭 Options Aggressiveness — CSP & Covered Calls")

# Load term structure + breadth
term = load_vix_term()
breadth = load_breadth_series()
breadth_trend = slope_of_ratio(breadth) if not breadth.empty else float("nan")

vix_last   = term["vix"].dropna().iloc[-1].item()   if not term["vix"].empty   else float("nan")
vix3m_last = term["vix3m"].dropna().iloc[-1].item() if not term["vix3m"].empty else float("nan")
baa_last   = extra["BaaSpread"].dropna().iloc[-1].item() if "BaaSpread" in extra and not extra["BaaSpread"].empty else float("nan")
spy_trend  = metrics["spy_trend"]

csp = csp_guidance(vix_last, vix3m_last, spy_trend, breadth_trend, baa_last)
cc  = cc_guidance(vix_last, vix3m_last, spy_trend, breadth_trend, baa_last)

left, right = st.columns(2)
with left:
    st.markdown("### 💵 Cash-Secured Puts")
    st.metric("CSP Score", f"{csp['score']:.0f}/100")
    st.write(f"**Stance:** {csp['level']}")
    st.write(f"**Suggested strike (Δ):** ~{csp['delta']:.2f}")
    st.write(f"**Suggested DTE:** {csp['dte']}")
    st.caption(csp["note"])

with right:
    st.markdown("### 📈 Covered Calls")
    st.metric("CC Score", f"{cc['score']:.0f}/100")
    st.write(f"**Stance:** {cc['level']}")
    st.write(f"**Suggested strike (Δ):** ~{cc['delta']:.2f}")
    st.write(f"**Suggested DTE:** {cc['dte']}")
    st.caption(cc["note"])

with left:
    st.markdown("**Inputs behind the gauge**")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("VIX (1M)", f"{vix_last:.1f}" if pd.notna(vix_last) else "N/A")
    g2.metric("VIX3M", f"{vix3m_last:.1f}" if pd.notna(vix3m_last) else "N/A")
    g3.metric("Term (3M-1M)", f"{(vix3m_last - vix_last):.1f}" if pd.notna(vix_last) and pd.notna(vix3m_last) else "N/A")
    g4.metric("Baa-10Y", f"{baa_last:.2f}" if pd.notna(baa_last) else "N/A")


# Small context row
st.caption(
    "Heuristics only. Δ is the **option delta** for target strikes. "
    "Use position sizing, risk limits, and exits (e.g., 50–75% profit takes)."
)

# =========================
# Time Series
# =========================
st.subheader("Time Series")

colA, colB = st.columns(2)
with colA:
    st.markdown("**M1 & M2 (Monthly, last value)**")
    m1m = m1.resample("ME").last()
    m2m = m2.resample("ME").last()
    st.line_chart(pd.DataFrame({"M1": m1m, "M2": m2m}))

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

# =========================
# MMF Flows (auto or upload)
# =========================
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
            mmf_df = pd.read_excel(mmf_file)  # pandas chooses engine
        mmf_msg = f"Loaded from upload: {mmf_file.name}"
except Exception as e:
    st.warning(f"MMF auto-download/parse failed: {e}. You can upload the file manually above.")

if mmf_df is not None:
    if "Date" not in mmf_df.columns:
        mmf_df.columns = [str(c).strip() for c in mmf_df.columns]
        date_col = next((c for c in mmf_df.columns if "date" in c.lower()), mmf_df.columns[0])
        mmf_df[date_col] = pd.to_datetime(mmf_df[date_col], errors="coerce")
        mmf_df = mmf_df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        mmf_df.index.name = "Date"

    st.caption(mmf_msg)

    num_cols = [c for c in mmf_df.columns if pd.api.types.is_numeric_dtype(mmf_df[c])]
    if not num_cols:
        st.warning("No numeric columns found to chart; showing table.")
        st.dataframe(mmf_df.tail(50), width="stretch")
    else:
        st.write("Detected numeric columns:", ", ".join(num_cols[:3]))
        st.line_chart(mmf_df[num_cols[:2]].tail(260))  # ~5 years weekly
else:
    st.info("Enable **Auto-download ICI MMF flows** or upload a file to see the chart.")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "Notes: Risk buckets are simplified heuristics. Tune thresholds in THRESHOLDS for your approach. "
    "Data from FRED (M1, M2, DGS10, DGS2, ICSA, USSLIND, BAA10YM) and Yahoo Finance (^GSPC, ^VIX, DXY)."
)
