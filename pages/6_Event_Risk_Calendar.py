from __future__ import annotations

import pandas as pd
import streamlit as st

from app_context import prepare_page
from compute.csp_attractiveness import csp_attractiveness, covered_call_suggestion
from compute.event_risk import build_event_risk_table


def render_event_calendar() -> None:
    st.markdown("## Event Risk Calendar")
    st.caption(
        "Surface earnings, dividends, and macro releases that could destabilize a wheel position. "
        "Probabilities model a stressed implied-vol regime heading into each catalyst."
    )

    raw_symbol = st.text_input("Ticker", value="AAPL", key="event_calendar_symbol")
    symbols = [s.strip().upper() for s in raw_symbol.split(",") if s.strip()]
    symbol = symbols[0] if symbols else ""
    if len(symbols) > 1:
        st.caption(f"Multiple tickers detected; focusing on **{symbol}**. Re-run with a single ticker for clarity.")

    settings_col1, settings_col2, settings_col3, settings_col4 = st.columns(4)
    with settings_col1:
        target_dte = st.slider("Target DTE", 7, 60, 30, step=1, key="event_calendar_dte")
    with settings_col2:
        target_delta = st.slider("Target put delta", 0.05, 0.40, 0.20, step=0.01, key="event_calendar_delta")
    with settings_col3:
        target_call_delta = st.slider(
            "Target call delta", 0.05, 0.40, 0.25, step=0.01, key="event_calendar_call_delta"
        )
    with settings_col4:
        horizon_days = st.slider("Horizon (days)", 15, 120, 60, step=5, key="event_calendar_horizon")

    stress_multiplier = st.slider(
        "Stress IV multiplier", 1.0, 2.0, 1.35, step=0.05, help="Scales the base ATM IV to approximate event vol expansion."
    )

    if not symbol:
        st.info("Enter a ticker to build the event calendar.")
        return

    result = csp_attractiveness(symbol, target_dte=target_dte, target_delta=target_delta)
    if not result.get("ok"):
        st.warning(result.get("msg", "Unable to score ticker."))
        return

    cc_result = covered_call_suggestion(
        symbol, target_dte=target_dte, target_delta=target_call_delta
    )

    call_strike_raw = cc_result.get("strike", float("nan"))
    call_strike_value = float(call_strike_raw) if cc_result.get("ok") and pd.notna(call_strike_raw) else float("nan")
    call_delta_raw = cc_result.get("delta", float("nan"))

    meta_row1 = st.columns(3)
    meta_row1[0].metric("Spot", f"{result.get('price', float('nan')):.2f}")
    meta_row1[1].metric(
        "Put strike",
        f"{result.get('strike', float('nan')):.2f}" if pd.notna(result.get("strike")) else "N/A",
    )
    meta_row1[2].metric(
        "Call strike",
        f"{call_strike_value:.2f}" if pd.notna(call_strike_value) else "N/A",
    )

    meta_row2 = st.columns(4)
    meta_row2[0].metric("Put |Δ|", f"{result.get('delta', float('nan')):.2f}" if pd.notna(result.get("delta")) else "N/A")
    meta_row2[1].metric(
        "Call Δ",
        f"{call_delta_raw:.2f}" if cc_result.get("ok") and pd.notna(call_delta_raw) else "N/A",
    )
    meta_row2[2].metric("Suggested expiry", result.get("expiry", "N/A"))
    meta_row2[3].metric(
        "ATM IV",
        f"{result.get('IV_atm_%', float('nan')):.1f}%" if pd.notna(result.get("IV_atm_%")) else "N/A",
    )

    event_df = build_event_risk_table(
        symbol=symbol,
        put_strike=float(result.get("strike", float("nan"))),
        call_strike=call_strike_value,
        expiry=str(result.get("expiry", "")),
        spot=float(result.get("price", float("nan"))),
        atm_iv_pct=result.get("IV_atm_%"),
        horizon_days=horizon_days,
        stress_multiplier=stress_multiplier,
    )

    st.divider()
    st.markdown("### Upcoming Catalysts")

    if event_df.empty:
        st.success("No flagged events within the selected horizon. Continue monitoring as new data arrives.")
        return

    warning_rows = event_df[event_df["Warning"] == "⚠️"] if "Warning" in event_df.columns else event_df.iloc[0:0]
    if not warning_rows.empty:
        top_risk = warning_rows.iloc[0]
        focus = top_risk.get("Risk Focus", "Put") or "Put"
        prob_col = "Prob Hit Put %" if focus == "Put" else "Prob Hit Call %"
        prob_val = top_risk.get(prob_col, float("nan"))
        prob_text = f"{prob_val:.1f}%" if pd.notna(prob_val) else "N/A"
        st.warning(
            f"{warning_rows.shape[0]} event(s) inside your expiry look risky. "
            f"Nearest: **{top_risk['Event']}** on {top_risk['Date'].date()} "
            f"({top_risk['Risk']} {focus.lower()} risk, {prob_text} breach probability)."
        )
    else:
        st.caption("No elevated risk inside your expiry. Still review macro releases beyond expiry for context.")

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

    column_config: dict[str, st.ColumnConfig] = {}
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
        "Probabilities assume normally distributed returns and do not account for gap risk beyond ±1σ. "
        "Layer in position sizing and hedges when catalysts cluster near your CSP or covered-call strikes."
    )


def main() -> None:
    prepare_page()
    render_event_calendar()


if __name__ == "__main__":
    main()
