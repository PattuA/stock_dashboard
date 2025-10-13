from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from compute.csp_attractiveness import csp_attractiveness, covered_call_suggestion


@dataclass
class WheelReadinessRow:
    ticker: str
    price: float | None
    csp_score: float | None
    csp_level: str | None
    csp_strike: float | None
    csp_delta: float | None
    cc_score: float | None
    cc_strike: float | None
    cc_delta: float | None
    cc_yield_pct: float | None
    cc_spread_pct: float | None
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "Ticker": self.ticker,
            "Price": self.price,
            "CSP Score": self.csp_score,
            "CSP Level": self.csp_level,
            "CSP Strike": self.csp_strike,
            "CSP |Delta|": self.csp_delta,
            "CC Score": self.cc_score,
            "CC Strike": self.cc_strike,
            "CC Delta": self.cc_delta,
            "CC Yield %": self.cc_yield_pct,
            "CC Spread %": self.cc_spread_pct,
            "Note": self.note or "",
        }


def _safe_round(value: float | None, digits: int) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _covered_call_score(price: float | None, mid: float | None, spread_pct: float | None, otm_pct: float | None) -> float | None:
    if price is None or not np.isfinite(price) or price <= 0:
        return None
    yield_component = 0.0
    if mid is not None and np.isfinite(mid) and mid > 0:
        yield_component = float(np.clip((mid / price) / 0.02, 0.0, 1.0))

    spread_component = 0.5
    if spread_pct is not None and np.isfinite(spread_pct):
        spread_component = float(1.0 - np.clip(spread_pct / 8.0, 0.0, 1.0))

    otm_component = 0.5
    if otm_pct is not None and np.isfinite(otm_pct):
        otm_component = float(np.clip(otm_pct / 5.0, 0.0, 1.0))

    score = 100.0 * (0.5 * yield_component + 0.3 * spread_component + 0.2 * otm_component)
    return round(score, 1)


def evaluate_wheel_readiness(
    tickers: Iterable[str],
    target_dte: int = 30,
    csp_delta: float = 0.20,
    cc_delta: float = 0.25,
) -> pd.DataFrame:
    rows: List[WheelReadinessRow] = []
    for symbol in tickers:
        clean = symbol.strip().upper()
        if not clean:
            continue

        csp = csp_attractiveness(clean, target_dte=target_dte, target_delta=csp_delta)
        cc = covered_call_suggestion(clean, target_dte=target_dte, target_delta=cc_delta)

        csp_ok = bool(csp.get("ok"))
        cc_ok = bool(cc.get("ok"))

        note_parts: List[str] = []
        if not csp_ok:
            note_parts.append(csp.get("msg", "CSP unavailable"))
        if not cc_ok:
            note_parts.append(cc.get("msg", "CC unavailable"))

        price = None
        if csp_ok:
            price = csp.get("price")
        elif cc_ok:
            price = cc.get("price")

        cc_score = None
        cc_yield = None
        cc_spread = None
        cc_delta_value = None
        cc_strike = None
        if cc_ok:
            price_val = cc.get("price")
            cc_mid = cc.get("mid")
            cc_spread = _safe_round(cc.get("spread_%"), 2)
            cc_delta_value = _safe_round(cc.get("delta"), 2)
            cc_strike = _safe_round(cc.get("strike"), 2)
            otm_pct = cc.get("%OTM")
            cc_score = _covered_call_score(price_val, cc_mid, cc.get("spread_%"), otm_pct)
            if price_val and np.isfinite(price_val) and cc_mid and np.isfinite(cc_mid):
                cc_yield = round(float(cc_mid) / float(price_val) * 100.0, 2)

        rows.append(
            WheelReadinessRow(
                ticker=clean,
                price=_safe_round(price, 2),
                csp_score=_safe_round(csp.get("score_total"), 1) if csp_ok else None,
                csp_level=csp.get("stance") if csp_ok else None,
                csp_strike=_safe_round(csp.get("strike"), 2) if csp_ok else None,
                csp_delta=_safe_round(csp.get("delta"), 2) if csp_ok else None,
                cc_score=cc_score,
                cc_strike=cc_strike,
                cc_delta=cc_delta_value,
                cc_yield_pct=cc_yield,
                cc_spread_pct=cc_spread,
                note="; ".join(note_parts) if note_parts else None,
            )
        )

    df = pd.DataFrame([row.as_dict() for row in rows])
    return df
