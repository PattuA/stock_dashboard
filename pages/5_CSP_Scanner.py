from __future__ import annotations

import streamlit as st

from app_context import prepare_page
from panels.csp_scanner_panel import render_csp_scanner


def main() -> None:
    prepare_page()

    st.page_link("app.py", label="Back to Overview")
    st.page_link("pages/4_MMF_Flows.py", label="Prev: MMF Flows")
    st.markdown("---")

    render_csp_scanner()


if __name__ == "__main__":
    main()
