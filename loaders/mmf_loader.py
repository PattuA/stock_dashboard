"""
ICI MMF flows auto-download and parse (with flexible sheet/column detection).
"""

import io, requests, pandas as pd, streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from config import ICI_MMF_PAGE

@st.cache_data(ttl=60*60*24, show_spinner=True)
def fetch_latest_ici_xls_url() -> str:
    r = requests.get(ICI_MMF_PAGE, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".xls"):
            return href if href.startswith("http") else urljoin(ICI_MMF_PAGE, href)
    return ""

@st.cache_data(ttl=60*60*24, show_spinner=True)
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
            if df.empty or df.shape[1] < 2: continue
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols: continue
            def pref(c):
                lc = c.lower()
                return (("flow" in lc or "net" in lc) * 2) + (("asset" in lc or "total" in lc)*1)
            num_cols.sort(key=pref, reverse=True)
            out = df[[date_col] + num_cols].copy().set_index(date_col)
            out.index.name = "Date"
            return out
    raise RuntimeError("Could not parse any sheet in the ICI workbook.")
