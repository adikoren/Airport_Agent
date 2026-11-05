"""Renovation Profitability Score (ARCHITECTURE §6.2–§6.4).

RPS = 100 × max(0, growth_share) × C
  growth_share: FAA TAF forecast growth over the horizon (fallback 2%/year if the FAA has no forecast)
  C: average of the 4 constraint signals, each as a percentile rank among US reference airports
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from airport_agent.settings import get_settings, load_config
from airport_agent.tools import metrics as m
from airport_agent.tools import public_store as ps
from airport_agent.tools.metrics import MetricValue

SIGNALS = ["load_factor", "peak_saturation", "taxi_out_excess", "taxi_in_excess"]


# ---------------------------------------------------------------- growth (step 1)

@dataclass
class Growth:
    growth_share: float
    base_year: int
    target_year: int
    source: str
    edition: str | None
    base_enplanements: float | None = None
    target_enplanements: float | None = None
    caveats: list[str] = field(default_factory=list)


def growth(faa_id: str | None, horizon: int, period: ps.Period, taf: pd.DataFrame | None = None) -> Growth:
    cfg = load_config("scoring")["growth"]
    taf = ps.taf() if taf is None else taf
    base, target = period.fiscal_year, period.fiscal_year + horizon
    rows = taf[taf.locid == faa_id].set_index("fiscal_year") if faa_id else taf.iloc[0:0]
    if base in rows.index and target in rows.index and rows.loc[base, "enplanements"] > 0:
        b, t = float(rows.loc[base, "enplanements"]), float(rows.loc[target, "enplanements"])
        return Growth(t / b - 1, base, target, "FAA TAF", str(rows.loc[base, "edition"]), b, t)
    rate = cfg["fallback_growth_per_year"]
    return Growth((1 + rate) ** horizon - 1, base, target, "fallback", None,
                  caveats=[f"fallback growth: the FAA has no forecast for this airport, so "
                           f"{rate:.0%} a year is used"])


# ---------------------------------------------------------------- constraint (step 3)

@lru_cache
def signal_reference() -> pd.DataFrame:
    """Reference values per signal (built by scripts/calibrate_signal_scales.py)."""
    return pd.read_parquet(get_settings().reference_dir / "signal_reference.parquet")


def reference_values(signal: str, ref: pd.DataFrame | None = None) -> np.ndarray:
    ref = signal_reference() if ref is None else ref
    return ref.loc[ref.signal == signal, "value"].to_numpy(dtype=float)


def weights(user_weights: dict[str, float] | None = None) -> dict[str, float]:
    w = dict(load_config("scoring")["constraint"]["weights"])
    if user_weights:
        if any(v < 0 for v in user_weights.values()):
            raise ValueError("weights must be non-negative")
        unknown = set(user_weights) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown signals: {sorted(unknown)}")
        w.update(user_weights)
    total = sum(w.values())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return {k: v / total for k, v in w.items()}


@dataclass
class Constraint:
    C: float | None
    percentiles: dict[str, float | None]
    raw: dict[str, float | None]
    caveats: list[str] = field(default_factory=list)


def constraint(signals: dict[str, MetricValue], ref: pd.DataFrame | None = None,
               user_weights: dict[str, float] | None = None) -> Constraint:
    w = weights(user_weights)
    pct, raw, caveats = {}, {}, []
    for s in SIGNALS:
        v = signals[s].value
        raw[s] = v
        pct[s] = None if v is None else m.percentile_rank(v, reference_values(s, ref))
        if v is None:
            caveats.append(f"constraint signal missing: {s.replace('_', ' ')} ({signals[s].missing_reason}); "
                           f"C is the average of the remaining signals")
    used = {s: w[s] for s in SIGNALS if pct[s] is not None and w[s] > 0}
    if not used:
        return Constraint(None, pct, raw, caveats)
    total = sum(used.values())
    C = sum(pct[s] * used[s] / total for s in used)
    return Constraint(float(C), pct, raw, caveats)


# ---------------------------------------------------------------- range (§6.4): bootstrap of days for C

def constraint_range(airport_id: int, period: ps.Period, lf_pct: float | None, available: set[str],
                     ref: pd.DataFrame | None = None,
                     user_weights: dict[str, float] | None = None) -> tuple[float, float] | None:
    """5th–95th percentile of C when the airport's days are resampled.

    Each airport's maximum hour is held at its full-sample value; days are resampled with replacement.
    Only the signals in `available` (the ones used for the airport's C) are resampled.
    """
    cfg = load_config("scoring")
    rcfg, ccfg = cfg["range"], cfg["constraint"]
    cells = ps.hourly_cells()
    w = weights(user_weights)
    per_day = {}
    for kind, s, n, sig in (("departures", "taxi_out_sum", "taxi_out_n", "taxi_out_excess"),
                            ("arrivals", "taxi_in_sum", "taxi_in_n", "taxi_in_excess")):
        c = m._cells(cells, airport_id, period, kind)
        if c.empty:
            return None
        cap = m.max_hour(c, kind)
        util = (c[kind] / cap).clip(upper=1.5)
        busy, quiet = util >= ccfg["busy_hour_min_utilization"], util <= ccfg["quiet_hour_max_utilization"]
        parts = pd.DataFrame({"date": c.date, "bs": c[s].where(busy, 0), "bn": c[n].where(busy, 0),
                              "qs": c[s].where(quiet, 0), "qn": c[n].where(quiet, 0)})
        day = parts.groupby("date").sum()                 # per-day totals in busy and quiet hours
        if kind == "departures":
            day["peak"] = c.groupby("date").departures.max() / cap
        per_day[sig] = day
    days = per_day["taxi_out_excess"].join(per_day["taxi_in_excess"].add_suffix("_in"), how="inner")
    if len(days) < 2:
        return None
    rng = np.random.default_rng(rcfg["bootstrap_seed"])
    refs = {s: reference_values(s, ref) for s in SIGNALS}
    arr = days.to_numpy(dtype=float)
    cols = {c: i for i, c in enumerate(days.columns)}
    Cs = []
    for _ in range(rcfg["bootstrap_resamples"]):
        smp = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        k = len(arr)
        vals = {
            "peak_saturation": smp[cols["peak"]] / k,
            "taxi_out_excess": smp[cols["bs"]] / max(smp[cols["bn"]], 1) - smp[cols["qs"]] / max(smp[cols["qn"]], 1),
            "taxi_in_excess": smp[cols["bs_in"]] / max(smp[cols["bn_in"]], 1) - smp[cols["qs_in"]] / max(smp[cols["qn_in"]], 1),
        }
        p = {s: m.percentile_rank(v, refs[s]) for s, v in vals.items() if s in available}
        if lf_pct is not None:
            p["load_factor"] = lf_pct
        used = {s: w[s] for s in p if w[s] > 0}
        Cs.append(sum(p[s] * used[s] for s in used) / sum(used.values()))
    lo, hi = rcfg["interval"]
    return float(np.quantile(Cs, lo)), float(np.quantile(Cs, hi))


# ---------------------------------------------------------------- drivers (no RPS)

def drivers(signals: dict[str, MetricValue], faa_id: str | None, passengers_now: float | None,
            period: ps.Period, horizon: int, user_weights: dict[str, float] | None = None) -> dict:
    """What drives an airport's position: constraint C and its signals, forecast growth, size.

    Used for comparisons and explanations. It does not compute RPS, so a question that only asks
    why an airport is constrained, or to compare two airports, does not score anything.
    """
    con = constraint(signals, user_weights=user_weights)
    g = growth(faa_id, horizon, period)
    extra = passengers_now * max(0.0, g.growth_share) if passengers_now is not None else None
    sig_out = {s: {"value": signals[s].value, "percentile": con.percentiles[s], "unit": signals[s].unit,
                   "source": signals[s].source, "missing_reason": signals[s].missing_reason} for s in SIGNALS}
    return {"C": con.C, "percentiles": con.percentiles, "signals": sig_out,
            "growth_share": g.growth_share, "growth_source": g.source,
            "growth_years": (g.base_year, g.target_year), "taf_edition": g.edition,
            "passengers_now": passengers_now, "extra_passengers": extra,
            "caveats": list(con.caveats) + list(g.caveats)}


# ---------------------------------------------------------------- airport score (steps 1–4)

@dataclass
class AirportScore:
    airport_id: int
    iata: str
    name: str
    state: str | None
    hub_size: str | None
    scored: bool
    not_scored_reason: str | None
    rps: float | None
    rps_low: float | None
    rps_high: float | None
    growth_share: float | None
    growth_source: str | None
    growth_years: tuple[int, int] | None
    taf_edition: str | None
    passengers_now: float | None
    extra_passengers: float | None
    C: float | None
    C_range: tuple[float, float] | None
    signals: dict[str, dict]
    international_share: float | None
    caveats: list[str]
    rank: int | None = None
    rank_value: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_signals(airport_id: int, period: ps.Period) -> dict[str, MetricValue]:
    cells, t = ps.hourly_cells(), ps.t100_monthly()
    return {
        "load_factor": m.load_factor(t, airport_id, period),
        "peak_saturation": m.peak_saturation(cells, airport_id, period),
        "taxi_out_excess": m.taxi_out_excess(cells, airport_id, period),
        "taxi_in_excess": m.taxi_in_excess(cells, airport_id, period),
    }


def score_airport(airport_id: int, period: ps.Period | None = None, horizon: int | None = None,
                  user_weights: dict[str, float] | None = None, with_range: bool = True,
                  signals: dict[str, MetricValue] | None = None) -> AirportScore:
    cfg = load_config("scoring")
    period = period or ps.default_period()
    horizon = horizon or cfg["horizon"]["default_years"]
    if horizon not in cfg["horizon"]["allowed_years"]:
        raise ValueError(f"horizon must be one of {cfg['horizon']['allowed_years']} years")
    ref_row = ps.airports().set_index("airport_id").loc[airport_id]
    signals = signals or compute_signals(airport_id, period)
    pax = m.passengers(ps.t100_monthly(), airport_id, period)
    intl = m.international_share(ps.t100_monthly(), airport_id, period)
    con = constraint(signals, user_weights=user_weights)
    g = growth(ref_row.faa_id, horizon, period)
    caveats = list(con.caveats) + list(g.caveats)
    sig_out = {s: {"value": signals[s].value, "percentile": con.percentiles[s], "unit": signals[s].unit,
                   "source": signals[s].source, "missing_reason": signals[s].missing_reason} for s in SIGNALS}
    base = dict(airport_id=int(airport_id), iata=ref_row.iata, name=ref_row["name"],
                state=None if pd.isna(ref_row.state) else ref_row.state,
                hub_size=None if pd.isna(ref_row.hub_size) else ref_row.hub_size,
                growth_share=g.growth_share, growth_source=g.source, growth_years=(g.base_year, g.target_year),
                taf_edition=g.edition, passengers_now=pax.value, signals=sig_out,
                international_share=intl.value, C=con.C)
    if signals["load_factor"].value is None or con.C is None:
        reason = signals["load_factor"].missing_reason or "no constraint signal available"
        return AirportScore(**base, scored=False, not_scored_reason=f"load factor missing: {reason}", rps=None,
                            rps_low=None, rps_high=None, extra_passengers=None, C_range=None, caveats=caveats)
    gs = max(0.0, g.growth_share)
    rps = 100 * gs * con.C
    available = {s for s in SIGNALS if con.percentiles[s] is not None}
    C_range = constraint_range(airport_id, period, con.percentiles["load_factor"], available,
                               user_weights=user_weights) if with_range else None
    if with_range and C_range is None:
        caveats.append("range not available: not enough days of hourly data to resample")
    if with_range:
        caveats.append("growth range not available yet: only the constraint part of the range is measured")
    extra = gs * pax.value if pax.value is not None else None
    return AirportScore(**base, scored=True, not_scored_reason=None, rps=rps,
                        rps_low=100 * gs * C_range[0] if C_range else None,
                        rps_high=100 * gs * C_range[1] if C_range else None,
                        extra_passengers=extra, C_range=C_range, caveats=caveats)


def measure(s: AirportScore, by: str, extra: dict[str, dict[str, float | None]] | None = None) -> float | None:
    """The value an airport is ranked by (ARCHITECTURE §6.2; 'rps' is the default)."""
    if by == "rps":
        return s.rps if s.scored and s.rps and s.rps > 0 else None
    if by == "constraint":
        return s.C
    if by == "passengers":
        return s.passengers_now
    if by == "growth_share":
        return s.growth_share
    if by == "international_share":
        return s.international_share
    if by in SIGNALS:
        return s.signals[by]["value"]
    return (extra or {}).get(s.iata, {}).get(by)


def rank(scores: list[AirportScore], by: str = "rps", order: str = "highest",
         extra: dict[str, dict[str, float | None]] | None = None) -> list[AirportScore]:
    """Sort airports by one measure. Airports with no value come last, unranked."""
    for s in scores:
        s.rank_value = measure(s, by, extra)
    rankable = [s for s in scores if s.rank_value is not None]
    rest = [s for s in scores if s.rank_value is None]
    rankable.sort(key=lambda s: s.rank_value, reverse=(order != "lowest"))
    for i, s in enumerate(rankable, 1):
        s.rank = i
    rest.sort(key=lambda s: (s.C or 0), reverse=True)
    return rankable + rest
