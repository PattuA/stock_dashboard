from __future__ import annotations

import streamlit as st

from app_context import prepare_page
from panels.timeseries_panel import render_base_charts


def main() -> None:
    ctx = prepare_page()

    st.page_link("app.py", label="Back to Overview")
    st.page_link("pages/2_Macro_Forecast.py", label="Prev: Macro Forecast")
    st.markdown("---")

    render_base_charts(ctx.m1, ctx.m2, ctx.y10, ctx.spx, ctx.vix)

    st.page_link("pages/4_MMF_Flows.py", label="Next: MMF Flows")


if __name__ == "__main__":
    main()
