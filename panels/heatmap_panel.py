import streamlit as st

from compute.risk_scoring import score_panel


def render_heatmap(m1, m2, y10, spx, vix):
    st.subheader("Risk Heat Map")
    heat_df, metrics = score_panel(m1, m2, y10, spx, vix)
    st.dataframe(heat_df, width="stretch", hide_index=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("10Y", f"{metrics['y10']:.2f}%")
    c2.metric("M2 YoY", f"{metrics['m2_yoy']:.2f}%")
    c3.metric("M1 YoY", f"{metrics['m1_yoy']:.2f}%")
    c4.metric("S&P Trend (50d vs 200d)", f"{metrics['spy_trend']:.2f}%")
    c5.metric("VIX", f"{metrics['vix']:.2f}")
    st.markdown("---")
    return metrics
