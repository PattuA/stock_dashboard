"""
FRED client / series loaders.
Reads FRED_API_KEY from .env and exposes cached loaders.
"""

import os, re
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fredapi import Fred

from config import FRED_SERIES, FRED_SERIES_EXTRA

# Resolve .env next to app.py
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

API_KEY = os.getenv("FRED_API_KEY", "")

if not re.fullmatch(r"[a-z0-9]{32}", API_KEY or ""):
    # fail early inside app import; app.py will show a nicer message
    FRED = None
else:
    FRED = Fred(api_key=API_KEY)

@st.cache_data(ttl=60*30, show_spinner=False)
def load_fred_series(series_id: str) -> pd.Series:
    if FRED is None:
        raise RuntimeError("FRED API key missing/invalid in .env")
    s = FRED.get_series(series_id)
    return s.dropna()

@st.cache_data(ttl=60*30, show_spinner=False)
def load_extra_series() -> dict:
    """Returns dict[str, Series] for 2Y, Claims, LEI, BaaSpread."""
    if FRED is None:
        raise RuntimeError("FRED API key missing/invalid in .env")
    out = {}
    for name, sid in FRED_SERIES_EXTRA.items():
        try:
            out[name] = FRED.get_series(sid).dropna()
        except Exception as e:
            st.warning(f"Could not load {name} ({sid}): {e}")
    return out
