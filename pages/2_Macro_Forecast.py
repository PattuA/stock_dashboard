from __future__ import annotations

import streamlit as st

from app_context import prepare_page
from compute.risk_scoring import score_panel
from panels.forecast_panel import render_forecast_panel


def main() -> None:
    ctx = prepare_page()

    st.page_link("app.py", label="Back to Overview")
    st.page_link("pages/1_Risk_Heatmap.py", label="Prev: Risk Heatmap")
    st.markdown("---")

    _, metrics = score_panel(ctx.m1, ctx.m2, ctx.y10, ctx.spx, ctx.vix)
    render_forecast_panel(metrics, ctx.extra)

    st.page_link("pages/3_Core_Time_Series.py", label="Next: Core Time Series")


if __name__ == "__main__":
    main()
