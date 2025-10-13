from __future__ import annotations

import streamlit as st

from app_context import prepare_page
from panels.wheel_heatmap_panel import render_wheel_heatmap_panel


def main() -> None:
    ctx = prepare_page()

    st.page_link("app.py", label="Back to Overview")
    st.markdown("---")

    render_wheel_heatmap_panel(spy_trend=ctx.spy_trend, extra=ctx.extra)


if __name__ == "__main__":
    main()
