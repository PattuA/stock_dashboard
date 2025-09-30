from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from compute.risk_scoring import composite_risk_score
from loaders.fred_loader import load_extra_series
from loaders.yf_loader import load_dxy


def _latest(series: Optional[pd.Series]) -> float:
    if series is None or series.empty:
        return float("nan")
    cleaned = series.dropna()
    if cleaned.empty:
        return float("nan")
    value = cleaned.iloc[-1]
    try:
        value = value.item()
    except AttributeError:
        pass
    except ValueError:
        pass
    return float(value)


def render_forecast_panel(metrics, extra: Optional[Dict[str, pd.Series]] = None) -> None:
    st.subheader("Macro Forecast Panel")
    extra = extra or load_extra_series()
    dxy = load_dxy()
    dxy_last = _latest(dxy)
    if not dxy.empty:
        st.line_chart(dxy)
    else:
        st.write("No DXY data available.")

    y10_last = metrics["y10"]
    spy_trend = metrics["spy_trend"]
    y2_last = _latest(extra.get("2Y") if extra else None)
    claims_last = _latest(extra.get("Claims") if extra else None)
    lei_last = _latest(extra.get("LEI") if extra else None)
    spread_last = _latest(extra.get("BaaSpread") if extra else None)

    risk_score = composite_risk_score(
        y10_last,
        y2_last,
        metrics["vix"],
        claims_last,
        spread_last,
        dxy_last,
        spy_trend,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("2Y Yield", f"{y2_last:.2f}%" if pd.notna(y2_last) else "N/A")
    c2.metric("Unemployment Claims", f"{claims_last:,.0f}" if pd.notna(claims_last) else "N/A")
    c3.metric("Baa Spread", f"{spread_last:.2f}" if pd.notna(spread_last) else "N/A")

    d1, d2, d3 = st.columns(3)
    d1.metric("LEI (Index)", f"{lei_last:.1f}" if pd.notna(lei_last) else "N/A")
    d2.metric("Dollar Index (DXY)", f"{dxy_last:.1f}" if pd.notna(dxy_last) else "N/A")
    d3.metric("Composite Risk Score", f"{risk_score:.1f}/100" if pd.notna(risk_score) else "N/A")

    st.markdown("**Macro Indicators (Recent)**")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**2Y Treasury Yield**")
        series_2y = extra.get("2Y") if extra else None
        if series_2y is not None and not series_2y.empty:
            st.line_chart(series_2y)
        else:
            st.write("No 2Y data available.")

        st.markdown("**Unemployment Claims**")
        series_claims = extra.get("Claims") if extra else None
        if series_claims is not None and not series_claims.empty:
            st.line_chart(series_claims.rename("Weekly Claims"))
        else:
            st.write("No claims data available.")

    with col_right:
        st.markdown("**Baa - 10Y Treasury Spread**")
        series_spread = extra.get("BaaSpread") if extra else None
        if series_spread is not None and not series_spread.empty:
            st.line_chart(series_spread.rename("Baa-10Y Spread"))
        else:
            st.write("No spread data available.")

        st.markdown("**US Dollar Index (DXY)**")
        if not dxy.empty:
            st.line_chart(dxy)
        else:
            st.write("No DXY data available.")

    st.markdown("---")
