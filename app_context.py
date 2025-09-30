from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import APP_TITLE, PAGE_ICON, FRED_SERIES, START_EQ_DATE
from dashboard import trend_slope_percent
from loaders import fred_loader
from loaders.fred_loader import load_extra_series, load_fred_series
from loaders.yf_loader import load_equity, load_vix


@dataclass
class AppContext:
    m1: pd.Series
    m2: pd.Series
    y10: pd.Series
    spx: pd.Series
    vix: pd.Series
    extra: dict[str, pd.Series]
    auto_mmf: bool
    mmf_file: st.runtime.uploaded_file_manager.UploadedFile | None
    spy_trend: float


def configure_page() -> None:
    """Set Streamlit page options."""
    st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide")


def fetch_fred_key() -> str:
    """Return the FRED API key from env vars or Streamlit secrets."""
    env_key = os.getenv("FRED_API_KEY", "")
    if env_key:
        return env_key
    try:
        return st.secrets["FRED_API_KEY"]  # type: ignore[index]
    except Exception:
        return ""


def mask_key(key: str, visible: int = 4) -> str:
    if not key:
        return "Missing"
    if len(key) <= visible * 2:
        return "*" * len(key)
    return f"{key[:visible]}****{key[-visible:]}"


def valid_fred_key(key: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]{32}", key or ""))


def build_sidebar(fred_key: str) -> Tuple[bool, st.runtime.uploaded_file_manager.UploadedFile | None]:
    with st.sidebar:
        st.subheader("Settings")
        st.caption("Mobile tip: use page links below to switch sections quickly.")
        key_display = mask_key(fred_key) if valid_fred_key(fred_key) else "Missing"
        st.write("FRED key:", key_display)
        st.divider()
        auto_mmf = st.toggle("Auto-download ICI MMF flows", value=True, key="auto_mmf_toggle")
        st.caption("If off, upload manually:")
        mmf_file = st.file_uploader(
            "ICI MMF flows (CSV/XLS/XLSX)",
            type=["csv", "xls", "xlsx"],
            key="mmf_file_uploader",
        )
    return auto_mmf, mmf_file


def ensure_fred_client(api_key: str) -> None:
    if not valid_fred_key(api_key):
        st.error(
            "FRED API key missing or invalid. "
            "Locally add `FRED_API_KEY=your32charlowercasekey` to `.env`, "
            "or populate it under Settings -> Secrets on Streamlit Cloud."
        )
        st.stop()
    os.environ["FRED_API_KEY"] = api_key
    fred_loader.configure_client(api_key)


@st.cache_data(show_spinner=False)
def _load_core_series_cached():
    m1 = load_fred_series(FRED_SERIES["M1"])
    m2 = load_fred_series(FRED_SERIES["M2"])
    y10 = load_fred_series(FRED_SERIES["10Y"])
    spx = load_equity("^GSPC", start=START_EQ_DATE)
    vix = load_vix(start=START_EQ_DATE)
    extra = load_extra_series()
    return m1, m2, y10, spx, vix, extra


def load_core_series():
    try:
        return _load_core_series_cached()
    except Exception as exc:
        st.error(f"FRED data load failed: {exc}")
        st.stop()


def prepare_page() -> AppContext:
    load_dotenv()
    configure_page()

    fred_key = fetch_fred_key()
    auto_mmf, mmf_file = build_sidebar(fred_key)

    ensure_fred_client(fred_key)

    m1, m2, y10, spx, vix, extra = load_core_series()
    spy_trend_value = trend_slope_percent(spx)

    return AppContext(
        m1=m1,
        m2=m2,
        y10=y10,
        spx=spx,
        vix=vix,
        extra=extra,
        auto_mmf=auto_mmf,
        mmf_file=mmf_file,
        spy_trend=spy_trend_value,
    )
