import streamlit as st, pandas as pd
from loaders.fred_loader import load_extra_series
from loaders.yf_loader import load_dxy
from compute.risk_scoring import composite_risk_score

def render_forecast_panel(metrics):
    st.subheader("🔮 Forecast Panel (Macro + Market Mix)")
    extra = load_extra_series()
    dxy = load_dxy()
    dxy_last = dxy.iloc[-1].item() if not dxy.empty else float("nan")
    if not dxy.empty:
        st.line_chart(dxy)  # already named "DXY"
    else:
        st.write("No DXY data")

    y10_last = metrics["y10"]
    spy_trend = metrics["spy_trend"]
    y2_last = extra["2Y"].dropna().iloc[-1].item() if "2Y" in extra and not extra["2Y"].empty else float("nan")
    claims_last = extra["Claims"].dropna().iloc[-1].item() if "Claims" in extra and not extra["Claims"].empty else float("nan")
    lei_last = extra["LEI"].dropna().iloc[-1].item() if "LEI" in extra and not extra["LEI"].empty else float("nan")
    spread_last = extra["BaaSpread"].dropna().iloc[-1].item() if "BaaSpread" in extra and not extra["BaaSpread"].empty else float("nan")

    risk_score = composite_risk_score(y10_last, y2_last, metrics["vix"], claims_last, spread_last, dxy_last, spy_trend)

    c1, c2, c3 = st.columns(3)
    c1.metric("2Y Yield", f"{y2_last:.2f}%" if pd.notna(y2_last) else "N/A")
    c2.metric("Unemployment Claims", f"{claims_last:,.0f}" if pd.notna(claims_last) else "N/A")
    c3.metric("Baa Spread", f"{spread_last:.2f}" if pd.notna(spread_last) else "N/A")

    d1, d2, d3 = st.columns(3)
    d1.metric("LEI (Index)", f"{lei_last:.1f}" if pd.notna(lei_last) else "N/A")
    d2.metric("Dollar Index (DXY)", f"{dxy_last:.1f}" if pd.notna(dxy_last) else "N/A")
    d3.metric("Composite Risk Score", f"{risk_score:.1f}/100" if pd.notna(risk_score) else "N/A")

    st.markdown("**Macro Indicators (Recent)**")
    colX, colY = st.columns(2)
    with colX:
        st.markdown("**2Y Treasury Yield**")
        if "2Y" in extra and not extra["2Y"].empty:
            st.line_chart(extra["2Y"])
        else:
            st.write("No 2Y data")

        st.markdown("**Unemployment Claims**")
        if "Claims" in extra and not extra["Claims"].empty:
            st.line_chart(extra["Claims"].rename("Weekly Claims"))
        else:
            st.write("No Claims data")

    with colY:
        st.markdown("**Baa – 10Y Treasury Spread**")
        if "BaaSpread" in extra and not extra["BaaSpread"].empty:
            st.line_chart(extra["BaaSpread"].rename("Baa-10Y Spread"))
        else:
            st.write("No Spread data")

        st.markdown("**US Dollar Index (DXY)**")
        if not dxy.empty:
            st.line_chart(dxy)
        else:
            st.write("No DXY data")

    st.markdown("---")
