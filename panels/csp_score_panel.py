from __future__ import annotations

import pandas as pd
import streamlit as st

from compute.csp_attractiveness import csp_attractiveness


def _fmt(value, pattern: str) -> str:
    return pattern.format(value) if pd.notna(value) else "N/A"


def render_csp_score_panel() -> None:
    st.subheader("CSP Attractiveness - Single Ticker")

    with st.expander("Model settings", expanded=True):
        raw = st.text_input("Ticker", value="AAPL", key="csp_score_ticker")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        symbol = symbols[0] if symbols else ""
        if len(symbols) > 1:
            st.caption(
                f"Multiple tickers detected: {', '.join(symbols)}. Scoring **{symbol}**. "
                "Use the scanner tab for multi-symbol runs."
            )

        target_dte = st.slider("Target DTE (days)", 7, 60, 30, step=1, key="csp_score_dte")
        target_delta = st.slider("Target put delta", 0.05, 0.40, 0.20, step=0.01, key="csp_score_delta")

    if not symbol:
        st.info("Enter a ticker to score.")
        return

    result = csp_attractiveness(symbol, target_dte=target_dte, target_delta=target_delta)
    if not result.get("ok"):
        st.warning(result.get("msg", "Unable to score ticker."))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", _fmt(result["score_total"], "{:.1f}/100"))
    c2.metric("Stance", result.get("stance", "N/A"))
    c3.metric("Price", _fmt(result["price"], "{:.2f}"))
    c4.metric("DTE target", f"{target_dte} days")

    st.markdown("**Suggested contract**")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Expiry", result.get("expiry", "N/A"))
    s2.metric("Strike", _fmt(result["strike"], "{:.2f}"))
    s3.metric("|Delta|", _fmt(result["delta"], "{:.2f}"))
    s4.metric("Mid", _fmt(result["mid"], "{:.2f}"))

    st.markdown("**Premium / Liquidity**")
    p1, p2, p3 = st.columns(3)
    p1.metric("ATM IV", _fmt(result["IV_atm_%"], "{:.1f}%"))
    p2.metric("Spread %", _fmt(result["spread_%"], "{:.2f}"))
    p3.metric("Open Interest", _fmt(result["OI"], "{:.0f}"))

    st.markdown("**Safety**")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("% OTM", _fmt(result["%OTM"], "{:.2f}%"))
    t2.metric("ATR(14) %", _fmt(result["ATR_%"], "{:.2f}"))
    t3.metric("Buffer (ATR multiples)", _fmt(result["buffer_ATR_x"], "{:.2f}"))
    t4.metric("vs 200d MA %", _fmt(result["MA200_%"], "{:.2f}"))

    st.markdown("**Market Regime**")
    r1, r2 = st.columns(2)
    r1.metric("VIX Contango (3M − 1M)", _fmt(result["contango"], "{:.2f}"))
    r2.metric("Breadth slope (RSP/SPY) %", _fmt(result["breadth_slope_%"], "{:.2f}"))

    st.markdown("**Component scores**")
    component_df = pd.DataFrame(
        [
            {
                "Premium": result["score_premium"],
                "Liquidity": result["score_liquidity"],
                "Safety": result["score_safety"],
                "Regime": result["score_regime"],
            }
        ]
    ).T.rename(columns={0: "Score"})
    st.bar_chart(component_df, width="stretch")

    st.caption(
        "Scoring weights: Premium 40%, Liquidity 20%, Safety 25%, Regime 15%. "
        "Safety blends OTM buffer, ATR, and distance to the 200-day moving average. "
        "Regime combines VIX term structure and breadth slope."
    )
