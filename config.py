"""
Global configuration for the Market Risk Dashboard.
Edit thresholds here to tune the traffic-light model.
"""

APP_TITLE = "📊 U.S. Market Risk Dashboard"
START_EQ_DATE = "2018-01-01"  # history window for equities

# Base FRED series
FRED_SERIES = {
    "M1": "M1SL",
    "M2": "M2SL",
    "10Y": "DGS10",  # 10-Year Treasury Yield (%)
}

# Extra macro (FRED)
FRED_SERIES_EXTRA = {
    "2Y": "DGS2",           # 2-Year Treasury yield
    "Claims": "ICSA",       # Initial unemployment claims (weekly)
    "LEI": "USSLIND",       # Leading Economic Index
    "BaaSpread": "BAA10YM", # Moody's Baa - 10Y Treasury spread
}

ICI_MMF_PAGE = "https://www.ici.org/research/stats/mmf"

# Heat map thresholds (tweak to taste)
THRESHOLDS = {
    "10Y": {"green_max": 3.75, "yellow_max": 4.25},       # % yield
    "M2_yoy": {"green_max": 5.0, "yellow_max": 9.0},      # % YoY
    "M1_yoy": {"green_max": 5.0, "yellow_max": 9.0},      # % YoY
    "SPY_trend": {"green_min": 0.0, "yellow_min": -2.0},  # % (50d-200d)/200d
    "VIX": {"green_max": 16.0, "yellow_max": 22.0},       # level
}
