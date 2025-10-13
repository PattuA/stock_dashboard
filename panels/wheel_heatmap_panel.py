from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from compute.metrics import slope_of_ratio
from compute.wheel_readiness import evaluate_wheel_readiness
from compute.options_guidance import cc_guidance, csp_guidance
from loaders.yf_loader import load_breadth_series, load_vix_term


def _latest_value(obj: pd.Series | pd.DataFrame | float | int | None) -> float:
    if obj is None:
        return float("nan")
    if isinstance(obj, (float, int)):
        return float(obj)
    series: pd.Series
    if isinstance(obj, pd.DataFrame):
        numeric = obj.select_dtypes(include=[np.number])
        if numeric.empty:
            return float("nan")
        series = numeric.iloc[:, 0]
    else:
        series = pd.to_numeric(obj, errors="coerce")
    series = series.dropna()
    if series.empty:
        return float("nan")
    value = series.iloc[-1]
    if isinstance(value, (pd.Series, pd.DataFrame)):
        try:
            value = value.squeeze()
        except Exception:
            return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _environment_guidance(spy_trend: float, extra: dict[str, pd.Series] | None) -> dict[str, dict]:
    term = load_vix_term()
    breadth = load_breadth_series()
    breadth_trend = slope_of_ratio(breadth) if breadth is not None and not breadth.empty else float("nan")

    vix_last = _latest_value(term.get("vix"))
    vix3m_last = _latest_value(term.get("vix3m"))
    baa_last = float("nan")
    if extra and "BaaSpread" in extra:
        baa_last = _latest_value(extra["BaaSpread"])

    csp_env = csp_guidance(vix_last, vix3m_last, spy_trend, breadth_trend, baa_last)
    cc_env = cc_guidance(vix_last, vix3m_last, spy_trend, breadth_trend, baa_last)
    return {"csp": csp_env, "cc": cc_env}


def _render_heatmap(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No tickers evaluated. Adjust settings or provide a watchlist.")
        return

    melt_df = df.melt(
        id_vars=["Ticker"],
        value_vars=["CSP Score", "CC Score"],
        var_name="Strategy",
        value_name="Score",
    )
    melt_df = melt_df.dropna(subset=["Score"])
    if melt_df.empty:
        st.info("No valid scores were returned for the selected universe.")
        return

    chart = (
        alt.Chart(melt_df)
        .mark_rect()
        .encode(
            x=alt.X("Strategy:N", title="Wheel Leg"),
            y=alt.Y("Ticker:N", sort="-x"),
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(domain=[0, 100], scheme="redyellowgreen"),
                legend=alt.Legend(title="Score"),
            ),
            tooltip=[
                alt.Tooltip("Ticker:N"),
                alt.Tooltip("Strategy:N"),
                alt.Tooltip("Score:Q", format=".1f"),
            ],
        )
        .properties(height=28 * len(df), width="container")
    )

    text = (
        alt.Chart(melt_df)
        .mark_text(color="black")
        .encode(
            x="Strategy:N",
            y=alt.Y("Ticker:N", sort="-x"),
            text=alt.Text("Score:Q", format=".0f"),
        )
    )
    st.altair_chart(chart + text, use_container_width=True)


def render_wheel_heatmap_panel(spy_trend: float, extra: dict[str, pd.Series] | None) -> None:
    st.subheader("Wheel Readiness Heatmap")
    st.caption("Contrast CSP vs covered-call setups across your watchlist with the current market regime.")

    default_watchlist = (
        "SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMD, TSLA, META, GOOGL, AMZN, "
        "JPM, BAC, XOM, KO, PEP, WMT, COST"
    )

    with st.expander("Settings", expanded=True):
        watchlist = st.text_area(
            "Watchlist (one per line or comma separated)",
            value=default_watchlist,
            key="wheel_heatmap_watchlist",
            height=100,
        )
        target_dte = st.slider("Target DTE", 7, 60, 30, step=1, key="wheel_heatmap_dte")
        csp_delta = st.slider("Target CSP delta", 0.05, 0.40, 0.20, step=0.01, key="wheel_heatmap_csp_delta")
        cc_delta = st.slider("Target CC delta", 0.05, 0.50, 0.25, step=0.01, key="wheel_heatmap_cc_delta")

    separators = [",", ";", "\n", "\t"]
    for sep in separators[1:]:
        watchlist = watchlist.replace(sep, separators[0])
    tickers = [ticker.strip().upper() for ticker in watchlist.split(separators[0]) if ticker.strip()]

    with st.spinner("Evaluating wheel readiness..."):
        df = evaluate_wheel_readiness(tickers, target_dte=target_dte, csp_delta=csp_delta, cc_delta=cc_delta)

    env = _environment_guidance(spy_trend, extra)
    env_cols = st.columns(2)
    env_cols[0].metric(
        "CSP Stance",
        f"{env['csp']['score']}/100",
        help=env["csp"]["note"],
        delta=env["csp"]["level"],
    )
    env_cols[1].metric(
        "CC Stance",
        f"{env['cc']['score']}/100",
        help=env["cc"]["note"],
        delta=env["cc"]["level"],
    )

    st.markdown("### Heatmap")
    _render_heatmap(df)

    st.markdown("### Detail Table")
    display_cols = [
        "Ticker",
        "Price",
        "CSP Score",
        "CSP Level",
        "CSP Strike",
        "CSP |Delta|",
        "CC Score",
        "CC Strike",
        "CC Delta",
        "CC Yield %",
        "CC Spread %",
        "Note",
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    st.caption(
        "Scores are heuristics only: CSP score mirrors the single-ticker model, while CC score weighs premium yield, "
        "bid/ask spread, and percent OTM. Use in concert with position sizing, catalyst checks, and risk limits."
    )
