from __future__ import annotations

import streamlit as st

from app_context import prepare_page
from panels.heatmap_panel import render_heatmap


def main() -> None:
    ctx = prepare_page()

    st.page_link("app.py", label="Back to Overview")
    st.markdown("---")

    render_heatmap(ctx.m1, ctx.m2, ctx.y10, ctx.spx, ctx.vix)

    st.page_link("pages/2_Macro_Forecast.py", label="Next: Macro Forecast")


if __name__ == "__main__":
    main()
