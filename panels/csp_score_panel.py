from __future__ import annotations

import pandas as pd
import streamlit as st

from compute.csp_attractiveness import csp_attractiveness, covered_call_suggestion
from compute.event_risk import build_event_risk_table


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

    cc_suggestion = covered_call_suggestion(
        symbol, target_dte=target_dte, target_delta=min(max(target_delta, 0.05), 0.40)
    )
    call_strike_raw = cc_suggestion.get("strike", float("nan"))
    call_strike_value = float(call_strike_raw) if cc_suggestion.get("ok") and pd.notna(call_strike_raw) else float("nan")

    event_df = build_event_risk_table(
        symbol=symbol,
        put_strike=float(result.get("strike", float("nan"))),
        call_strike=call_strike_value,
        expiry=str(result.get("expiry", "")),
        spot=float(result.get("price", float("nan"))),
        atm_iv_pct=result.get("IV_atm_%"),
    )

    summary_cols = st.columns(5)
    summary_cols[0].metric("Score", _fmt(result["score_total"], "{:.1f}/100"))
    summary_cols[1].metric("Stance", result.get("stance", "N/A"))
    summary_cols[2].metric("Price", _fmt(result["price"], "{:.2f}"))
    summary_cols[3].metric("DTE target", f"{target_dte} days")
    summary_cols[4].metric(
        "CC Strike",
        f"{call_strike_value:.2f}" if pd.notna(call_strike_value) else "N/A",
    )

    st.divider()

    csp_tab, cc_tab = st.tabs(["CSP", "Covered Calls"])

    with csp_tab:
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

    with cc_tab:
        if not cc_suggestion.get("ok"):
            st.info("Unable to fetch covered-call suggestion for this expiry/delta.")
        else:
            top_row = st.columns(3)
            top_row[0].metric("Expiry", cc_suggestion.get("expiry", "N/A"))
            top_row[1].metric("Strike", _fmt(cc_suggestion.get("strike"), "{:.2f}"))
            top_row[2].metric("Delta", _fmt(cc_suggestion.get("delta"), "{:.2f}"))

            mid_row = st.columns(3)
            mid_row[0].metric("Mid", _fmt(cc_suggestion.get("mid"), "{:.2f}"))
            mid_row[1].metric("Spread %", _fmt(cc_suggestion.get("spread_%"), "{:.2f}"))
            mid_row[2].metric("Open Interest", _fmt(cc_suggestion.get("OI"), "{:.0f}"))

            bottom_row = st.columns(2)
            bottom_row[0].metric("% OTM", _fmt(cc_suggestion.get("%OTM"), "{:.2f}%"))
            bottom_row[1].metric("Underlying Price", _fmt(cc_suggestion.get("price"), "{:.2f}"))

    st.divider()

    st.markdown("#### Event Risk Calendar")
    if event_df.empty:
        st.caption("No high-impact earnings, dividends, or macro releases inside the next 60 days.")
    else:
        warning_rows = event_df[event_df["Warning"] == "⚠️"] if "Warning" in event_df.columns else event_df.iloc[0:0]
        if not warning_rows.empty:
            soonest = warning_rows.iloc[0]
            risk_focus = soonest.get("Risk Focus", "Put") or "Put"
            prob_col = "Prob Hit Put %" if risk_focus == "Put" else "Prob Hit Call %"
            prob_value = soonest.get(prob_col, float("nan"))
            prob_display = f"{prob_value:.1f}%" if pd.notna(prob_value) else "N/A"
            st.warning(
                f"{warning_rows.shape[0]} event(s) flagged. "
                f"Nearest: **{soonest['Event']}** on {soonest['Date'].date()} "
                f"({soonest['Risk']} {risk_focus.lower()} risk, "
                f"breach prob {prob_display})."
            )

        display_cols = [
            "Date",
            "Event",
            "Type",
            "Risk",
            "Risk Focus",
            "Prob Hit Put %",
            "Prob Hit Call %",
            "±1σ Move %",
            "Within Expiry",
            "Detail",
            "Note",
            "URL",
        ]
        available_cols = [col for col in display_cols if col in event_df.columns]
        column_config = {}
        if "Date" in available_cols:
            column_config["Date"] = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm", timezone="UTC")
        if "Prob Hit Put %" in available_cols:
            column_config["Prob Hit Put %"] = st.column_config.NumberColumn(format="%.1f%%")
        if "Prob Hit Call %" in available_cols:
            column_config["Prob Hit Call %"] = st.column_config.NumberColumn(format="%.1f%%")
        if "±1σ Move %" in available_cols:
            column_config["±1σ Move %"] = st.column_config.NumberColumn(format="%.2f%%")
        if "URL" in available_cols:
            column_config["URL"] = st.column_config.LinkColumn("Source")

        st.dataframe(
            event_df[available_cols],
            hide_index=True,
            use_container_width=True,
            column_config=column_config or None,
        )
        st.caption(
            "Probabilities assume a stressed IV multiplier of 1.35× the current ATM level. "
            "Assessments reference the suggested CSP and covered-call strikes and ignore position sizing."
        )

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
