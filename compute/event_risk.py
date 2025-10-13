from __future__ import annotations

"""
Build a probability-aware event calendar for option wheel management.
Merges symbol-specific events (earnings/dividends) with macro releases and
estimates the probability that the suggested CSP or covered-call strikes are challenged around each event.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Optional

import numpy as np
import pandas as pd

from loaders.event_loader import load_symbol_events


@dataclass
class EventRiskRow:
    name: str
    date: datetime
    type: str
    source: str
    detail: str
    days_out: float
    within_expiry: bool
    stress_iv_pct: float
    expected_move_pct: float
    breach_probability: float
    risk_bucket: str
    warning: bool
    note: str
    put_probability: float | None = None
    call_probability: float | None = None
    risk_side: str | None = None
    url: str | None = None

    def as_dict(self) -> dict:
        return {
            "Event": self.name,
            "Date": self.date,
            "Type": self.type,
            "Source": self.source,
            "Detail": self.detail,
            "Days Out": round(self.days_out, 1),
            "Within Expiry": "Yes" if self.within_expiry else "Post-expiry",
            "Stress IV %": round(self.stress_iv_pct, 1),
            "±1σ Move %": round(self.expected_move_pct, 2),
            "Prob Hit Put %": round(self.put_probability * 100.0, 1)
            if self.put_probability is not None and np.isfinite(self.put_probability)
            else np.nan,
            "Prob Hit Call %": round(self.call_probability * 100.0, 1)
            if self.call_probability is not None and np.isfinite(self.call_probability)
            else np.nan,
            "Risk": self.risk_bucket,
            "Risk Focus": self.risk_side if self.risk_side else "",
            "Warning": "⚠️" if self.warning else "",
            "Note": self.note,
            "URL": self.url or "",
        }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _norm_tail_probability(z: float) -> float:
    """Probability a standard normal surpasses +z in absolute value direction."""
    if z <= 0:
        return 0.5
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _risk_bucket(prob: float) -> str:
    if prob >= 0.45:
        return "High"
    if prob >= 0.25:
        return "Elevated"
    return "Low"


def build_event_risk_table(
    symbol: str,
    put_strike: float,
    call_strike: float | None,
    expiry: str,
    spot: float,
    atm_iv_pct: Optional[float],
    horizon_days: int = 60,
    stress_multiplier: float = 1.35,
) -> pd.DataFrame:
    """
    Returns DataFrame summarizing risk for upcoming events.

    Parameters
    ----------
    symbol : str
        Underlying ticker (used for earnings/dividend lookup).
    put_strike : float
        Suggested CSP strike price.
    call_strike : float | None
        Suggested covered-call strike price (optional).
    expiry : str
        Expiry string in '%Y-%m-%d' format.
    spot : float
        Current underlying price.
    atm_iv_pct : float | None
        ATM implied volatility in percent (e.g., 22.5).
    horizon_days : int
        How far ahead to pull events.
    stress_multiplier : float
        Multiplier applied to base IV to model pre-event vol expansion.
    """
    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        expiry_dt = None

    events_df = load_symbol_events(symbol, horizon_days=horizon_days)
    if events_df.empty or not np.isfinite(spot) or spot <= 0 or not np.isfinite(put_strike):
        return pd.DataFrame(columns=["Event", "Date", "Type", "Source", "Detail"])

    base_iv = float(atm_iv_pct) / 100.0 if atm_iv_pct and np.isfinite(atm_iv_pct) else 0.25
    base_iv = max(base_iv, 0.05)
    stress_iv = min(base_iv * stress_multiplier, 1.0)

    put_distance_pct = np.nan
    call_distance_pct = np.nan
    if np.isfinite(put_strike):
        put_distance_pct = (spot - put_strike) / spot
        put_distance_pct = max(put_distance_pct, 0.0)
    if call_strike is not None and np.isfinite(call_strike):
        call_distance_pct = (call_strike / spot) - 1.0
        call_distance_pct = max(call_distance_pct, 0.0)
    now = _now_utc()

    rows: list[EventRiskRow] = []
    for _, row in events_df.iterrows():
        event_dt = row["date"].to_pydatetime() if isinstance(row["date"], pd.Timestamp) else row["date"]
        if not isinstance(event_dt, datetime):
            continue
        event_dt = event_dt.astimezone(timezone.utc)
        delta_days = (event_dt - now).total_seconds() / 86400.0
        if delta_days < -2:
            continue
        trading_days = max(delta_days * (252.0 / 365.0), 1.0 / 252.0)
        expected_move_pct = stress_iv * math.sqrt(trading_days)

        sigma = max(expected_move_pct, 1e-4)

        if np.isfinite(put_distance_pct):
            if put_distance_pct <= 0:
                put_probability = 1.0
            else:
                z_put = put_distance_pct / sigma
                put_probability = _norm_tail_probability(z_put)
        else:
            put_probability = float("nan")

        if np.isfinite(call_distance_pct):
            if call_distance_pct <= 0:
                call_probability = 1.0
            else:
                z_call = call_distance_pct / sigma
                call_probability = _norm_tail_probability(z_call)
        else:
            call_probability = float("nan")

        prob_candidates = []
        if np.isfinite(put_probability):
            prob_candidates.append(("Put", put_probability))
        if np.isfinite(call_probability):
            prob_candidates.append(("Call", call_probability))
        if prob_candidates:
            risk_side, risk_prob = max(prob_candidates, key=lambda item: item[1])
        else:
            risk_side, risk_prob = ("N/A", 0.0)

        risk_bucket = _risk_bucket(risk_prob) if risk_prob > 0 else "Low"
        within_expiry = bool(expiry_dt and event_dt <= expiry_dt)
        warning = within_expiry and risk_prob >= 0.25

        notes = []
        if np.isfinite(put_distance_pct) and np.isfinite(put_probability):
            notes.append(f"Put dist {put_distance_pct*100:.1f}% (prob {put_probability*100:.1f}%)")
        else:
            notes.append("Put strike unavailable")
        if np.isfinite(call_distance_pct) and np.isfinite(call_probability):
            notes.append(f"Call dist {call_distance_pct*100:.1f}% (prob {call_probability*100:.1f}%)")
        else:
            notes.append("Call strike unavailable")
        notes.append(f"Stress ±1σ: {expected_move_pct*100:.1f}%")
        note = " | ".join(notes)
        rows.append(
            EventRiskRow(
                name=row["name"],
                date=event_dt,
                type=row.get("type", ""),
                source=row.get("source", ""),
                detail=row.get("detail", ""),
                days_out=max(delta_days, 0.0),
                within_expiry=within_expiry,
                stress_iv_pct=stress_iv * 100.0,
                expected_move_pct=expected_move_pct * 100.0,
                breach_probability=risk_prob,
                put_probability=put_probability,
                call_probability=call_probability,
                risk_side=risk_side,
                risk_bucket=risk_bucket,
                warning=warning,
                note=note,
                url=row.get("url"),
            )
        )

    df = pd.DataFrame([r.as_dict() for r in rows])
    if df.empty:
        return df

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
