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

    summary_cols = st.columns(4)
    summary_cols[0].metric("Score", _fmt(result["score_total"], "{:.1f}/100"))
    summary_cols[1].metric("Stance", result.get("stance", "N/A"))
    summary_cols[2].metric("Price", _fmt(result["price"], "{:.2f}"))
    summary_cols[3].metric("DTE target", f"{target_dte} days")

    st.divider()

    left_section, right_section = st.columns(2)

    with left_section:
        st.markdown("#### Suggested Contract")
        contract_top = st.columns(2)
        contract_top[0].metric("Expiry", result.get("expiry", "N/A"))
        contract_top[1].metric("Strike", _fmt(result["strike"], "{:.2f}"))
        contract_bottom = st.columns(2)
        contract_bottom[0].metric("|Delta|", _fmt(result["delta"], "{:.2f}"))
        contract_bottom[1].metric("Mid", _fmt(result["mid"], "{:.2f}"))

        st.markdown("#### Premium & Liquidity")
        premium_cols = st.columns(3)
        premium_cols[0].metric("ATM IV", _fmt(result["IV_atm_%"], "{:.1f}%"))
        premium_cols[1].metric("Spread %", _fmt(result["spread_%"], "{:.2f}"))
        premium_cols[2].metric("Open Interest", _fmt(result["OI"], "{:.0f}"))

    with right_section:
        st.markdown("#### Safety")
        safety_top = st.columns(2)
        safety_top[0].metric("% OTM", _fmt(result["%OTM"], "{:.2f}%"))
        safety_top[1].metric("ATR(14) %", _fmt(result["ATR_%"], "{:.2f}"))
        safety_bottom = st.columns(2)
        safety_bottom[0].metric("Buffer (ATR multiples)", _fmt(result["buffer_ATR_x"], "{:.2f}"))
        safety_bottom[1].metric("vs 200d MA %", _fmt(result["MA200_%"], "{:.2f}"))

        st.markdown("#### Market Regime")
        regime_cols = st.columns(2)
        regime_cols[0].metric("VIX Contango (3M - 1M)", _fmt(result["contango"], "{:.2f}"))
        regime_cols[1].metric("Breadth slope (RSP/SPY) %", _fmt(result["breadth_slope_%"], "{:.2f}"))

    st.divider()

    st.markdown("#### Component Scores")
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
