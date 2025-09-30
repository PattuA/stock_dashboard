from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from compute.metrics import slope_of_ratio
from compute.options_guidance import cc_guidance, csp_guidance
from loaders.yf_loader import load_breadth_series, load_vix_term


def _latest(series: pd.Series) -> float:
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


def render_options_panel(extra: Dict[str, pd.Series], spy_trend: float) -> None:
    st.subheader("Options Aggressiveness - CSP & Covered Calls")

    term_structure = load_vix_term()
    breadth = load_breadth_series()
    breadth_trend = slope_of_ratio(breadth) if breadth is not None and not breadth.empty else float("nan")

    vix_last = _latest(term_structure["vix"])
    vix3m_last = _latest(term_structure["vix3m"])
    baa_last = _latest(extra.get("BaaSpread")) if extra else float("nan")

    csp = csp_guidance(vix_last, vix3m_last, spy_trend, breadth_trend, baa_last)
    cc = cc_guidance(vix_last, vix3m_last, spy_trend, breadth_trend, baa_last)

    left, right = st.columns(2)
    with right:
        st.markdown("### Cash-Secured Puts")
        st.metric("CSP Score", f"{csp['score']:.0f}/100")
        st.write(f"**Stance:** {csp['level']}")
        st.write(f"**Suggested strike (delta):** ~{csp['delta']:.2f}")
        st.write(f"**Suggested DTE:** {csp['dte']}")
        st.caption(csp["note"])

    with left:
        st.markdown("### Covered Calls")
        st.metric("CC Score", f"{cc['score']:.0f}/100")
        st.write(f"**Stance:** {cc['level']}")
        st.write(f"**Suggested strike (delta):** ~{cc['delta']:.2f}")
        st.write(f"**Suggested DTE:** {cc['dte']}")
        st.caption(cc["note"])

    st.markdown("**Inputs behind the gauge**")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("VIX (1M)", f"{vix_last:.1f}" if pd.notna(vix_last) else "N/A")
    g2.metric("VIX 3M", f"{vix3m_last:.1f}" if pd.notna(vix3m_last) else "N/A")
    g3.metric(
        "Term (3M - 1M)",
        f"{(vix3m_last - vix_last):.1f}" if pd.notna(vix_last) and pd.notna(vix3m_last) else "N/A",
    )
    g4.metric("Baa - 10Y", f"{baa_last:.2f}" if pd.notna(baa_last) else "N/A")

    st.caption(
        "Heuristics only. Manage position sizing, risk limits, and exits (for example, 50-75% profit targets)."
    )
    st.markdown("---")

