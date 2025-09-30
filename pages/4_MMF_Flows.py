from __future__ import annotations

import streamlit as st

from app_context import prepare_page
from panels.timeseries_panel import render_mmf_section


def main() -> None:
    ctx = prepare_page()

    st.page_link("app.py", label="Back to Overview")
    st.page_link("pages/3_Core_Time_Series.py", label="Prev: Core Time Series")
    st.markdown("---")

    render_mmf_section(auto_mmf=ctx.auto_mmf, mmf_file=ctx.mmf_file)

    st.page_link("pages/5_CSP_Scanner.py", label="Next: CSP Scanner")


if __name__ == "__main__":
    main()
