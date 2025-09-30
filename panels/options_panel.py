import pandas as pd, streamlit as st
from loaders.yf_loader import load_vix_term, load_breadth_series
from compute.metrics import slope_of_ratio
from compute.options_guidance import csp_guidance, cc_guidance

def render_options_panel(extra, spy_trend):
    st.subheader("🧭 Options Aggressiveness — CSP & Covered Calls")

    term = load_vix_term()
    breadth = load_breadth_series()
    breadth_trend = slope_of_ratio(breadth) if not breadth.empty else float("nan")

    vix_last   = term["vix"].dropna().iloc[-1].item()   if not term["vix"].empty   else float("nan")
    vix3m_last = term["vix3m"].dropna().iloc[-1].item() if not term["vix3m"].empty else float("nan")
    baa_last   = extra["BaaSpread"].dropna().iloc[-1].item() if "BaaSpread" in extra and not extra["BaaSpread"].empty else float("nan")

    csp = csp_guidance(vix_last, vix3m_last, spy_trend, breadth_trend, baa_last)
    cc  = cc_guidance(vix_last, vix3m_last, spy_trend, breadth_trend, baa_last)

    left, right = st.columns(2)
    with right:
        st.markdown("### 💵 Cash-Secured Puts")
        st.metric("CSP Score", f"{csp['score']:.0f}/100")
        st.write(f"**Stance:** {csp['level']}")
        st.write(f"**Suggested strike (Δ):** ~{csp['delta']:.2f}")
        st.write(f"**Suggested DTE:** {csp['dte']}")
        st.caption(csp["note"])

    with left:
        st.markdown("### 📈 Covered Calls")
        st.metric("CC Score", f"{cc['score']:.0f}/100")
        st.write(f"**Stance:** {cc['level']}")
        st.write(f"**Suggested strike (Δ):** ~{cc['delta']:.2f}")
        st.write(f"**Suggested DTE:** {cc['dte']}")
        st.caption(cc["note"])

    st.markdown("**Inputs behind the gauge**")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("VIX (1M)", f"{vix_last:.1f}" if pd.notna(vix_last) else "N/A")
    g2.metric("VIX3M", f"{vix3m_last:.1f}" if pd.notna(vix3m_last) else "N/A")
    g3.metric("Term (3M-1M)", f"{(vix3m_last - vix_last):.1f}" if pd.notna(vix_last) and pd.notna(vix3m_last) else "N/A")
    g4.metric("Baa-10Y", f"{baa_last:.2f}" if pd.notna(baa_last) else "N/A")

    st.caption("Heuristics only. Use position sizing, risk limits, and exits (e.g., 50–75% profit takes).")
    st.markdown("---")
