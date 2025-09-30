"""FRED client helpers and cached loaders."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fredapi import Fred

from config import FRED_SERIES_EXTRA

_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH)

_API_KEY: str | None = None
_FRED: Fred | None = None


def _is_valid(key: str | None) -> bool:
    return bool(key and re.fullmatch(r"[a-z0-9]{32}", key))


def configure_client(api_key: str | None) -> None:
    """Configure the module-level Fred client."""
    global _API_KEY, _FRED
    if not _is_valid(api_key):
        _API_KEY = None
        _FRED = None
        return
    if api_key == _API_KEY and _FRED is not None:
        return
    _API_KEY = api_key
    _FRED = Fred(api_key=api_key)


def _client() -> Fred:
    if _FRED is None:
        raise RuntimeError("FRED API key missing or invalid. Call configure_client() first.")
    return _FRED


configure_client(os.getenv("FRED_API_KEY", ""))


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_fred_series(series_id: str) -> pd.Series:
    client = _client()
    data = client.get_series(series_id)
    return data.dropna()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_extra_series() -> Dict[str, pd.Series]:
    client = _client()
    out: Dict[str, pd.Series] = {}
    for name, sid in FRED_SERIES_EXTRA.items():
        try:
            out[name] = client.get_series(sid).dropna()
        except Exception as exc:
            st.warning(f"Could not load {name} ({sid}): {exc}")
    return out
