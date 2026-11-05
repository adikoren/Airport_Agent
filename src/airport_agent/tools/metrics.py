"""Metric and signal formulas (ARCHITECTURE §6.3, §6.6). Pure functions on the processed data.

Every function returns a MetricValue: the value (or None) plus the reason when it's missing,
the period, the source and the number of flights behind it. Missing is never turned into 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from airport_agent.settings import load_config
from airport_agent.tools import checks
from airport_agent.tools.public_store import Period


@dataclass
class MetricValue:
    value: float | None
    unit: str
    source: str
    period: str
    n: int | None = None
    missing_reason: str | None = None
    caveats: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _missing(unit: str, source: str, period: Period, reason: str, n: int | None = None) -> MetricValue:
    return MetricValue(None, unit, source, period.label, n=n, missing_reason=reason)


def _in_period(df: pd.DataFrame, col: str, period: Period) -> pd.DataFrame:
    return df[(df[col] >= period.start) & (df[col] < period.end + pd.DateOffset(months=1))]


# ---------------------------------------------------------------- T-100 based

def t100_window(t100: pd.DataFrame, airport_id: int, period: Period) -> pd.DataFrame:
    return _in_period(t100[t100.airport_id == airport_id], "month", period)


def passengers(t100: pd.DataFrame, airport_id: int, period: Period) -> MetricValue:
    w = t100_window(t100, airport_id, period)
    if reason := checks.complete_months(w.month.nunique()):
        return _missing("passengers", "BTS T-100", period, reason)
    return MetricValue(float(w.passengers.sum()), "passengers", "BTS T-100", period.label)


def load_factor(t100: pd.DataFrame, airport_id: int, period: Period) -> MetricValue:
    """Σ passengers / Σ seats over the window (sum over sum)."""
    w = t100_window(t100, airport_id, period)
    if reason := checks.complete_months(w.month.nunique()):
        return _missing("share", "BTS T-100", period, reason)
    seats = w.seats.sum()
    if seats <= 0:
        return _missing("share", "BTS T-100", period, "no seats reported")
    lf = float(w.passengers.sum() / seats)
    if reason := checks.load_factor_possible(lf):
        return _missing("share", "BTS T-100", period, reason)
    return MetricValue(lf, "share", "BTS T-100", period.label)


def international_share(t100: pd.DataFrame, airport_id: int, period: Period) -> MetricValue:
    w = t100_window(t100, airport_id, period)
    if reason := checks.complete_months(w.month.nunique()):
        return _missing("share", "BTS T-100", period, reason)
    total = w.passengers.sum()
    if total <= 0:
        return _missing("share", "BTS T-100", period, "no passengers reported")
    return MetricValue(float(w.intl_passengers.sum() / total), "share", "BTS T-100", period.label)


# ---------------------------------------------------------------- hourly signals (BTS on-time)

def _cells(cells: pd.DataFrame, airport_id: int, period: Period, kind: str) -> pd.DataFrame:
    """Cells with at least one scheduled flight of this kind ('departures' or 'arrivals')."""
    c = cells[(cells.airport_id == airport_id) & (cells[kind] > 0)]
    return c[(c.date >= period.start) & (c.date < period.end + pd.DateOffset(months=1))]


def max_hour(c: pd.DataFrame, kind: str) -> float:
    q = load_config("scoring")["constraint"]["max_hour_percentile"]
    return float(c[kind].quantile(q))


def peak_saturation(cells: pd.DataFrame, airport_id: int, period: Period) -> MetricValue:
    """Average over days of the busiest hour's departures ÷ the airport's maximum hour."""
    c = _cells(cells, airport_id, period, "departures")
    if c.empty:
        return _missing("share", "BTS on-time", period, "no departures in the on-time data")
    cap = max_hour(c, "departures")
    daily_peak = c.groupby("date").departures.max()
    return MetricValue(float(daily_peak.mean() / cap), "share", "BTS on-time", period.label,
                       n=int(c.departures.sum()), detail={"max_hour": cap, "days": int(daily_peak.size)})


def busy_minus_quiet(cells: pd.DataFrame, airport_id: int, period: Period, kind: str) -> MetricValue:
    """Mean taxi time in busy hours − mean taxi time in quiet hours, same airport.

    kind='departures' → taxi-out (runway queue); kind='arrivals' → taxi-in (waiting for a gate).
    Means are sum over sum (flight-weighted). Both groups need enough flights.
    """
    cfg = load_config("scoring")["constraint"]
    s, n = ("taxi_out_sum", "taxi_out_n") if kind == "departures" else ("taxi_in_sum", "taxi_in_n")
    what = "taxi-out" if kind == "departures" else "taxi-in"
    c = _cells(cells, airport_id, period, kind)
    if c.empty:
        return _missing("minutes", "BTS on-time", period, f"no {kind} in the on-time data")
    cap = max_hour(c, kind)
    util = (c[kind] / cap).clip(upper=1.5)
    busy, quiet = c[util >= cfg["busy_hour_min_utilization"]], c[util <= cfg["quiet_hour_max_utilization"]]
    nb, nq = int(busy[n].sum()), int(quiet[n].sum())
    for group, count in (("busy hours", nb), ("quiet hours", nq)):
        if reason := checks.enough_flights(count, f"{what} in {group}"):
            return _missing("minutes", "BTS on-time", period, reason, n=count)
    busy_mean, quiet_mean = busy[s].sum() / nb, quiet[s].sum() / nq
    return MetricValue(float(busy_mean - quiet_mean), "minutes", "BTS on-time", period.label, n=nb + nq,
                       detail={"busy_mean": float(busy_mean), "quiet_mean": float(quiet_mean),
                               "busy_flights": nb, "quiet_flights": nq, "max_hour": cap})


def taxi_out_excess(cells: pd.DataFrame, airport_id: int, period: Period) -> MetricValue:
    return busy_minus_quiet(cells, airport_id, period, "departures")


def taxi_in_excess(cells: pd.DataFrame, airport_id: int, period: Period) -> MetricValue:
    return busy_minus_quiet(cells, airport_id, period, "arrivals")


# ---------------------------------------------------------------- information measures (BTS on-time)

def delay_and_cancellation(ontime_m: pd.DataFrame, airport_id: int, period: Period) -> dict[str, MetricValue]:
    w = _in_period(ontime_m[ontime_m.airport_id == airport_id], "month", period)
    out = {}
    if w.empty:
        for k in ("delay_rate", "cancellation_rate"):
            out[k] = _missing("share", "BTS on-time", period, "no flights in the on-time data")
        return out
    out["delay_rate"] = MetricValue(float(w.delayed15.sum() / max(w.delay_known.sum(), 1)), "share",
                                    "BTS on-time", period.label, n=int(w.delay_known.sum()))
    out["cancellation_rate"] = MetricValue(float(w.cancelled.sum() / w.flights.sum()), "share",
                                           "BTS on-time", period.label, n=int(w.flights.sum()))
    return out


def long_haul_bts(ontime_m: pd.DataFrame, airport_id: int, period: Period) -> MetricValue:
    """Backup long-haul share from BTS: domestic flights only (stated as a caveat)."""
    w = _in_period(ontime_m[ontime_m.airport_id == airport_id], "month", period)
    flights = int(w.flights.sum())
    if flights == 0:
        return _missing("share", "BTS on-time", period, "no flights in the on-time data")
    known = int(w.block_time_known.sum())
    return MetricValue(
        float(w.long_haul_time.sum() / max(known, 1)), "share", "BTS on-time", period.label, n=flights,
        caveats=["domestic flights only: the BTS on-time data has no international flights"],
        detail={"by_block_time": float(w.long_haul_time.sum() / max(known, 1)),
                "by_distance": float(w.long_haul_dist.sum() / flights), "flights": flights})


def long_haul_from_flights(flights: list[dict], period_label: str) -> MetricValue:
    """Long-haul share from OAG operating flights (list of dicts with block_minutes, miles, passenger)."""
    lh = load_config("scoring")["long_haul"]
    if not flights:
        return MetricValue(None, "share", "OAG", period_label, missing_reason="no flights returned by OAG")
    df = pd.DataFrame(flights)
    known = df.block_minutes.notna()
    by_time = float((df.loc[known, "block_minutes"] >= lh["block_time_minutes"]).mean()) if known.any() else None
    by_dist = float((df.miles >= lh["distance_miles"]).mean())
    pax = df[df.passenger]
    pax_known = pax.block_minutes.notna()
    detail = {
        "by_block_time": by_time, "by_distance": by_dist, "flights": len(df),
        "block_time_missing": int((~known).sum()),
        "passenger_flights": len(pax),
        "passenger_by_block_time": float((pax.loc[pax_known, "block_minutes"] >= lh["block_time_minutes"]).mean())
        if pax_known.any() else None,
        "passenger_by_distance": float((pax.miles >= lh["distance_miles"]).mean()) if len(pax) else None,
        "international_share": float(df.international.mean()),
    }
    value = by_time if lh["default"] == "block_time" else by_dist
    caveats = []
    if (~known).any():
        caveats.append(f"{int((~known).sum())} flights had no usable block time; the distance-based share covers them")
    return MetricValue(value, "share", "OAG", period_label, n=len(df), caveats=caveats, detail=detail)


def percentile_rank(value: float, reference: np.ndarray) -> float:
    """Share of reference values below `value`, ties counting half (ARCHITECTURE §6.3)."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        raise ValueError("empty reference")
    return float(((ref < value).sum() + 0.5 * (ref == value).sum()) / ref.size)
