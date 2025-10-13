from __future__ import annotations

import streamlit as st

from app_context import prepare_page
from panels.csp_score_panel import render_csp_score_panel
from panels.options_panel import render_options_panel


def render_navigation() -> None:
    st.markdown("### Explore Pages")
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("app.py", label="Option Aggressiveness")
        st.page_link("pages/1_Risk_Heatmap.py", label="Risk Heatmap")
        st.page_link("pages/2_Macro_Forecast.py", label="Macro Forecast")
    with col2:
        st.page_link("pages/3_Core_Time_Series.py", label="Core Time Series")
        st.page_link("pages/4_MMF_Flows.py", label="MMF Flows")
        st.page_link("pages/5_CSP_Scanner.py", label="CSP Scanner")


def main() -> None:
    ctx = prepare_page()

    st.markdown("## Option Aggressiveness (CSP / Covered Calls)")
    render_options_panel(ctx.extra, spy_trend=ctx.spy_trend)

    st.divider()
    render_navigation()

    st.divider()
    st.markdown(
        "Key options context appears first, with detailed sections available via the page links above. "
        "Data sources: FRED (M1, M2, DGS10, DGS2, ICSA, USSLIND, BAA10YM) and "
        "Yahoo Finance (^GSPC, ^VIX, DXY)."
    )

    st.markdown("### CSP Ticker Model")
    render_csp_score_panel()


if __name__ == "__main__":
    main()
