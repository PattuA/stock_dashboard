"""Shared configuration values for the Streamlit dashboard."""

from __future__ import annotations

APP_TITLE = "Market Risk & Options Dashboard"
PAGE_ICON = "MR"
START_EQ_DATE = "2018-01-01"

FRED_SERIES = {
    "M1": "M1SL",
    "M2": "M2SL",
    "10Y": "DGS10",
}

FRED_SERIES_EXTRA = {
    "2Y": "DGS2",
    "Claims": "ICSA",
    "LEI": "USSLIND",
    "BaaSpread": "BAA10YM",
}

ICI_MMF_PAGE = "https://www.ici.org/research/stats/mmf"

THRESHOLDS = {
    "10Y": {"green_max": 3.75, "yellow_max": 4.25},
    "M2_yoy": {"green_max": 5.0, "yellow_max": 9.0},
    "M1_yoy": {"green_max": 5.0, "yellow_max": 9.0},
    "SPY_trend": {"green_min": 0.0, "yellow_min": -2.0},
    "VIX": {"green_max": 16.0, "yellow_max": 22.0},
}
