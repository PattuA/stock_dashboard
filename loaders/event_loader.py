from __future__ import annotations

"""
Helpers to gather upcoming symbol-specific and macroeconomic events.
Parsed events are normalized to UTC datetimes to simplify downstream risk calcs.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Iterable, List

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html",
}


@dataclass
class EventRecord:
    name: str
    date: datetime
    type: str
    detail: str
    source: str
    url: str | None = None
    importance: int = 1

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "date": self.date,
            "type": self.type,
            "detail": self.detail,
            "source": self.source,
            "url": self.url,
            "importance": self.importance,
        }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(ts: pd.Timestamp | datetime) -> datetime:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def load_earnings_events(symbol: str, limit: int = 6) -> List[EventRecord]:
    events: List[EventRecord] = []
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.get_earnings_dates(limit=limit)
    except Exception:
        return events

    if df is None or df.empty:
        return events

    df = df.reset_index(names="event_date")
    cutoff = _now_utc() - timedelta(days=1)
    for _, row in df.iterrows():
        event_date = row.get("event_date")
        if not isinstance(event_date, (pd.Timestamp, datetime)):
            continue
        dt_utc = _ensure_utc(event_date)
        if dt_utc < cutoff:
            continue
        eps_est = row.get("EPS Estimate")
        detail_parts = []
        if pd.notna(eps_est):
            detail_parts.append(f"EPS est: {eps_est:.2f}")
        surprise = row.get("Surprise(%)")
        if pd.notna(surprise):
            detail_parts.append(f"Surprise: {surprise:.2f}%")
        detail = ", ".join(detail_parts) if detail_parts else "Upcoming earnings"
        events.append(
            EventRecord(
                name=f"{symbol.upper()} earnings",
                date=dt_utc,
                type="Earnings",
                detail=detail,
                source="Yahoo Finance",
                url=f"https://finance.yahoo.com/quote/{symbol.upper()}/analysis",
                importance=3,
            )
        )
    return events


def load_dividend_events(symbol: str) -> List[EventRecord]:
    events: List[EventRecord] = []
    try:
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends
    except Exception:
        return events

    if dividends is None or dividends.empty:
        return events

    dividends = dividends.sort_index()
    now = pd.Timestamp.utcnow()
    future = dividends[dividends.index >= now]
    series_to_process = future if not future.empty else dividends.tail(1)

    for idx, amount in series_to_process.iloc[:2].items():
        try:
            dt_utc = _ensure_utc(idx)
        except Exception:
            continue
        if dt_utc < _now_utc() - timedelta(days=1):
            continue
        detail = f"Dividend ${float(amount):.2f}"
        if future.empty:
            detail += " (estimated from trend)"
            dt_utc = dt_utc + timedelta(days=90)
        events.append(
            EventRecord(
                name=f"{symbol.upper()} dividend",
                date=dt_utc,
                type="Dividend",
                detail=detail,
                source="Yahoo Finance",
                url=f"https://finance.yahoo.com/quote/{symbol.upper()}/history?filter=dividends",
                importance=2,
            )
        )
    return events


def _parse_bls_table(url: str, title: str) -> Iterable[EventRecord]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return []

    # Schedule table is typically the last table on the page
    table = tables[-1]
    events: List[EventRecord] = []
    ny_tz = ZoneInfo("America/New_York")

    rows = table.find_all("tr")
    for row in rows[1:]:
        cols = [col.get_text(strip=True) for col in row.find_all("td")]
        if len(cols) < 2:
            continue
        reference_month, date_text = cols[0], cols[1]
        time_text = cols[2] if len(cols) >= 3 and cols[2] else "08:30 AM"
        dt_utc = _parse_bls_datetime(date_text, time_text, ny_tz)
        if dt_utc is None:
            continue
        detail = f"{title} ({reference_month})"
        events.append(
            EventRecord(
                name=title,
                date=dt_utc,
                type="Macro",
                detail=detail,
                source="Bureau of Labor Statistics",
                url=url,
                importance=3,
            )
        )
    return events


def _parse_bls_datetime(date_text: str, time_text: str, tzinfo: ZoneInfo) -> datetime | None:
    clean_date = date_text.replace(".", "")
    clean_date = clean_date.replace("Sept", "Sep")
    try:
        date_part = datetime.strptime(clean_date, "%b %d, %Y")
    except ValueError:
        return None

    clean_time = time_text.strip().upper().replace("ET", "")
    if not clean_time:
        clean_time = "08:30 AM"
    try:
        time_part = datetime.strptime(clean_time, "%I:%M %p").time()
    except ValueError:
        time_part = datetime.strptime("08:30 AM", "%I:%M %p").time()

    combined = datetime.combine(date_part.date(), time_part, tzinfo=tzinfo)
    return combined.astimezone(timezone.utc)


@lru_cache(maxsize=1)
def _cached_macro_events() -> pd.DataFrame:
    events: List[EventRecord] = []
    events.extend(_parse_bls_table("https://www.bls.gov/schedule/news_release/cpi.htm", "CPI release"))
    events.extend(
        _parse_bls_table("https://www.bls.gov/schedule/news_release/empsit.htm", "Employment Situation")
    )
    data = [event.as_dict() for event in events]
    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def load_macro_events(horizon_days: int = 60) -> pd.DataFrame:
    df = _cached_macro_events().copy()
    if df.empty:
        return df
    now = pd.Timestamp.utcnow()
    upper = now + pd.Timedelta(days=horizon_days)
    return df[(df["date"] >= now - pd.Timedelta(days=1)) & (df["date"] <= upper)]


def load_symbol_events(symbol: str, horizon_days: int = 60) -> pd.DataFrame:
    records: List[EventRecord] = []
    records.extend(load_earnings_events(symbol))
    records.extend(load_dividend_events(symbol))

    try:
        macro_df = load_macro_events(horizon_days)
        macro_records = [
            EventRecord(
                name=row["name"],
                date=row["date"].to_pydatetime(),
                type=row["type"],
                detail=row["detail"],
                source=row["source"],
                url=row.get("url"),
                importance=int(row.get("importance", 1)),
            )
            for _, row in macro_df.iterrows()
        ]
        records.extend(macro_records)
    except Exception:
        pass

    if not records:
        return pd.DataFrame(columns=["name", "date", "type", "detail", "source", "importance", "url"])

    df = pd.DataFrame([record.as_dict() for record in records])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    now = pd.Timestamp.utcnow()
    upper = now + pd.Timedelta(days=horizon_days)
    df = df[(df["date"] >= now - pd.Timedelta(days=1)) & (df["date"] <= upper)]
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
