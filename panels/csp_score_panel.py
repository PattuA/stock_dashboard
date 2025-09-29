# panels/csp_score_panel.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from compute.csp_attractiveness import csp_attractiveness

def render_csp_score_panel():
    st.subheader("🎯 CSP Attractiveness (Ticker Model)")

# panels/csp_score_panel.py  (only the input block changed)

    with st.expander("Model settings", expanded=True):
        raw = st.text_input("Ticker", value="AAPL", key="csp_score_ticker")
        # allow comma-separated input but use the first symbol
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        symbol = symbols[0] if symbols else ""
        if len(symbols) > 1:
            st.caption(f"Multiple tickers detected: {', '.join(symbols)}. Scoring **{symbol}**. (Use scanner for multi-symbol.)")

        target_dte = st.slider("Target DTE (days)", 7, 60, 30, step=1, key="csp_score_dte")
        target_delta = st.slider("Target put delta", 0.05, 0.40, 0.20, step=0.01, key="csp_score_delta")


    if not symbol:
        st.info("Enter a ticker to score.")
        return

    res = csp_attractiveness(symbol, target_dte=target_dte, target_delta=target_delta)

    if not res.get("ok"):
        st.warning(res.get("msg", "Unable to score ticker."))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{res['score_total']:.1f}/100")
    c2.metric("Stance", res["stance"])
    c3.metric("Price", f"{res['price']:.2f}")
    c4.metric("DTE target", f"{target_dte}d")

    st.markdown("**Suggested contract**")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Expiry", res["expiry"])
    s2.metric("Strike", f"{res['strike']:.2f}")
    s3.metric("Δ (abs)", f"{res['delta']:.2f}")
    s4.metric("Mid", f"{res['mid']:.2f}")

    st.markdown("**Premium / Liquidity**")
    l1, l2, l3 = st.columns(3)
    l1.metric("ATM IV", f"{res['IV_atm_%'] if res['IV_atm_%']==res['IV_atm_%'] else 'N/A'}%")
    l2.metric("Spread %", f"{res['spread_%'] if res['spread_%']==res['spread_%'] else 'N/A'}")
    l3.metric("Open Interest", f"{res['OI']}")

    st.markdown("**Safety**")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("% OTM", f"{res['%OTM'] if res['%OTM']==res['%OTM'] else 'N/A'}%")
    t2.metric("ATR(14) %", f"{res['ATR_%'] if res['ATR_%']==res['ATR_%'] else 'N/A'}")
    t3.metric("Buffer (×ATR)", f"{res['buffer_ATR_x'] if res['buffer_ATR_x']==res['buffer_ATR_x'] else 'N/A'}")
    t4.metric("vs 200d MA %", f"{res['MA200_%'] if res['MA200_%']==res['MA200_%'] else 'N/A'}")

    st.markdown("**Market Regime**")
    r1, r2 = st.columns(2)
    r1.metric("VIX Contango (3M-1M)", f"{res['contango'] if res['contango']==res['contango'] else 'N/A'}")
    r2.metric("Breadth slope (RSP/SPY) %", f"{res['breadth_slope_%'] if res['breadth_slope_%']==res['breadth_slope_%'] else 'N/A'}")

    st.markdown("**Component scores**")
    df = pd.DataFrame([{
        "Premium": res["score_premium"],
        "Liquidity": res["score_liquidity"],
        "Safety": res["score_safety"],
        "Regime": res["score_regime"],
    }]).T.rename(columns={0: "Score"})
    st.bar_chart(df, width="stretch")


    st.caption(
        "Scoring: Premium (40%), Liquidity (20%), Safety (25%), Regime (15). "
        "Safety blends OTM buffer vs ATR, ATR level, and distance to 200d. "
        "Regime blends VIX term contango and breadth slope. Heuristics only."
    )
