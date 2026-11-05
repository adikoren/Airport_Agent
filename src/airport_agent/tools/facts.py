"""Build the facts bundle (ARCHITECTURE §4.4 node 7): the only numbers Claude may quote.

Every number is rounded once, here, and formatted the way the answer should show it.
Assumption, scope and method text comes from config/answer_text.yaml.
"""

from __future__ import annotations

from airport_agent.settings import load_config
from airport_agent.tools.scoring import SIGNALS, AirportScore

# Labels say what each number is calculated from. They do not state what the number means: taxi values are a
# busy-minus-quiet difference (an indirect indicator), and peak saturation is a daily average, not a capacity figure.
LABEL = {"load_factor": "load factor (passengers ÷ seats, 12 months)",
         "peak_saturation": "peak saturation (busiest-hour departures ÷ the airport's maximum hour)",
         "taxi_out_excess": "cleaned taxi-out (busy-hour minus quiet-hour mean taxi-out)",
         "taxi_in_excess": "cleaned taxi-in (busy-hour minus quiet-hour mean taxi-in)"}

# Coverage of the passenger figure (BTS T-100 segment data, see metrics.passengers).
PASSENGER_SOURCE = ("BTS T-100 segment data: passengers boarding at this airport, carriers that file T-100, "
                    "domestic and international segments")
RANGE_MEANING = ("The RPS range varies only with the constraint C (resampled days). Forecast growth is held at its "
                 "FAA value, so the range does not show forecast uncertainty; the growth range is not measured yet.")


def pct(x: float | None, d: int = 1) -> str | None:
    return None if x is None else f"{x * 100:.{d}f}%"


def num(x: float | None, d: int = 1) -> str | None:
    return None if x is None else f"{x:.{d}f}"


def people(x: float | None) -> str | None:
    if x is None:
        return None
    return f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.0f}K" if x >= 1e4 else f"{x:,.0f}"


def _signal(s: str, v: dict) -> dict:
    val = v["value"]
    shown = None if val is None else (pct(val) if s in ("load_factor", "peak_saturation") else f"{val:+.1f} min")
    return {"signal": LABEL[s], "value": shown,
            "percentile": None if v["percentile"] is None else f"higher than {v['percentile'] * 100:.0f}% of US airports",
            "missing_reason": v["missing_reason"]}


def signals_measured(a) -> str:
    """'4 of 4', or e.g. '2 of 4 ⚑' when C rests on fewer signals (kept in the ranking, but flagged)."""
    sig = a.signals if hasattr(a, "signals") else a
    n = sum(1 for s in SIGNALS if sig[s]["value"] is not None)
    return f"{n} of {len(SIGNALS)}" + ("" if n == len(SIGNALS) else " ⚑")


def growth_status(source: str | None, share: float | None) -> str | None:
    """Missing forecast and non-positive growth are different facts; the answer must say which one applies."""
    if share is None:
        return None
    if source == "fallback":
        fb = load_config("scoring")["growth"]["fallback_growth_per_year"]
        return f"no FAA forecast: fallback growth of {pct(fb, 0)} a year is used (our assumption, not an FAA figure)"
    if share <= 0:
        return "FAA forecast shows no growth over the horizon (growth is not positive)"
    return "FAA forecast"


def airport_row(a: AirportScore) -> dict:
    row = {
        "rank": a.rank, "airport": f"{a.name} ({a.iata})", "iata": a.iata, "state": a.state,
        "faa_hub_size": a.hub_size,
        "RPS": num(a.rps), "RPS_range": None if a.rps_low is None else f"{num(a.rps_low)}–{num(a.rps_high)}",
        "forecast_growth": pct(a.growth_share),
        "forecast_status": growth_status(a.growth_source, a.growth_share),
        "growth_period": f"FY{a.growth_years[0]}–FY{a.growth_years[1]}" if a.growth_years else None,
        "growth_source": a.growth_source,
        "forecast_extra_passengers": people(a.extra_passengers),
        "passengers_last_12_months": people(a.passengers_now),
        "passengers_source": PASSENGER_SOURCE if a.passengers_now is not None else None,
        "C": num(a.C, 2), "C_range": None if a.C_range is None else f"{a.C_range[0]:.2f}–{a.C_range[1]:.2f}",
        "signals_measured": signals_measured(a),
        "constraint_signals": [_signal(s, a.signals[s]) for s in SIGNALS],
        "international_share": pct(a.international_share),
        "caveats": a.caveats,
    }
    if not a.scored:
        row["not_scored_reason"] = a.not_scored_reason
    elif not a.rps:      # RPS is 0: either growth is not positive, or C is 0. Say which.
        if a.growth_share is not None and a.growth_share <= 0:
            row["note"] = "FAA forecast shows no growth over the horizon, so RPS is 0"
        else:
            row["note"] = "constraint C is 0, so RPS is 0"
    return row


def assumptions(taf_edition: str | None) -> list[str]:
    t = load_config("answer_text")
    fb = load_config("scoring")["growth"]["fallback_growth_per_year"]
    return [x.format(taf_edition=taf_edition or "TAF", fallback=fb) for x in t["assumptions"]]


def common(period_label: str, horizon: int | None, scope_notes: list[str], taf_edition: str | None) -> dict:
    """Screening facts: they carry the data period, the assumptions and the investment disclaimer."""
    cfg = load_config("scoring")
    return {"answer_mode": "computed", "score_name": "Renovation Profitability Score (RPS)", "data_period": period_label,
            "data_window_months": cfg["checks"]["months_in_window"],
            "constraint_signal_count": len(cfg["constraint"]["signals"]), "horizon_years": horizon,
            "assumptions": assumptions(taf_edition), "scope_notes": scope_notes,
            "disclaimer": load_config("answer_text")["disclaimer"]}


MEASURES = {"rps": "Renovation Profitability Score (RPS)", "constraint": "constraint score C",
            "passengers": "passengers in the last 12 months", "growth_share": "FAA forecast growth",
            "load_factor": LABEL["load_factor"], "peak_saturation": LABEL["peak_saturation"],
            "taxi_out_excess": LABEL["taxi_out_excess"], "taxi_in_excess": LABEL["taxi_in_excess"],
            "international_share": "international passenger share", "delay_rate": "departure delay rate (≥ 15 min)",
            "cancellation_rate": "cancellation rate"}


def measure_text(by: str, v: float | None) -> str | None:
    if v is None:
        return None
    if by == "passengers":
        return people(v)
    if by in ("rps",):
        return num(v)
    if by == "constraint":
        return num(v, 2)
    if by in ("taxi_out_excess", "taxi_in_excess"):
        return f"{v:+.1f} min"
    return pct(v)


def ranking_facts(scores: list[AirportScore], region: str | None, period_label: str, horizon: int,
                  scope_notes: list[str], params: dict | None = None) -> dict:
    from airport_agent.tools.resolve import is_national
    params = params or {}
    by, order = params.get("rank_by", "rps"), params.get("order", "highest")
    ranked = [s for s in scores if s.rank]
    rest = [s for s in scores if not s.rank]
    shown_n = params.get("top_n") or (len(ranked) if len(ranked) <= 20 else 10)
    shown = ranked[:shown_n]
    rows = []
    for s in shown:
        row = airport_row(s)
        row["ranked_by_value"] = measure_text(by, s.rank_value)
        rows.append(row)
    edition = next((s.taf_edition for s in scores if s.taf_edition), None)
    scope = "all US airports with flight data" if is_national(region) else region
    out = {"question_type": "rank", "scope": scope, "region": None if is_national(region) else region,
           "ranked_by": MEASURES[by], "order": f"{order} first",
           "airports_in_scope": len(scores), "airports_ranked": len(ranked), "airports_shown": len(shown),
           "ranking": rows,
           "listed_not_ranked": [airport_row(s) for s in rest][:10],
           "listed_not_ranked_count": len(rest),
           "score_formula": "RPS = 100 × forecast growth × C",
           **common(period_label, horizon, scope_notes, edition)}
    flagged = [r["iata"] for r in rows if r["signals_measured"] and "⚑" in r["signals_measured"]]
    if flagged:
        out["flag_note"] = (f"⚑ {', '.join(flagged)}: C is based on fewer than {len(SIGNALS)} signals, because the "
                            "airport is too small to have both busy and quiet hours; compare it with care.")
    if len(shown) < len(ranked):
        out["scope_notes"] = list(out["scope_notes"]) + [f"showing the top {len(shown)} of {len(ranked)} ranked airports"]
    if any(r.get("RPS_range") for r in rows):
        out["range_meaning"] = RANGE_MEANING
    # Ranked but not displayed (outside the shown count). They are ranked, not "unranked".
    out["ranked_not_shown"] = [{"rank": s.rank, "airport": f"{s.name} ({s.iata})", "RPS": num(s.rps),
                                "note": airport_row(s).get("note")} for s in ranked[shown_n:]][:10]
    out["custom_weights"] = bool(params.get("weights"))
    return out


def essential_caveats(facts: dict) -> list[dict]:
    """Caveats an answer must state for this specific result, even in a focused follow-up.

    Each item has the text the answer should carry and the phrases that show it does (any one is enough).
    They come from the facts, so the number check and the fallback template use the same rules.
    """
    rows = facts.get("ranking") or facts.get("airports") or []
    out: list[dict] = []
    if any((r.get("forecast_status") or "").startswith("no FAA forecast") for r in rows):
        out.append({"text": "Some airports have no FAA forecast; their growth uses our fallback rate, not an FAA figure.",
                    "accept": ["fallback", "no FAA forecast", "not an FAA"]})
    if any("RPS is 0" in (r.get("note") or "") for r in rows):
        out.append({"text": "Airports with RPS 0 are listed with their reason: no growth in the FAA forecast, or a "
                            "constraint C of 0.",
                    "accept": ["RPS 0", "RPS is 0", "RPS of 0"]})
    if facts.get("flag_note"):
        out.append({"text": "Airports marked ⚑ rest on fewer than four constraint signals.",
                    "accept": ["⚑", "fewer than", "fewer signals"]})
    if facts.get("range_meaning"):
        out.append({"text": "The RPS range covers only the constraint C; forecast growth is held at its FAA value.",
                    "accept": ["constraint", "growth is held"]})
    h, default = facts.get("horizon_years"), load_config("scoring")["horizon"]["default_years"]
    if h is not None and h != default:
        out.append({"text": f"The horizon is {h} years, so the growth figures cover that period, not the default "
                            f"{default} years.",
                    "accept": [f"{h} year", f"{h}-year", "horizon"]})
    if facts.get("custom_weights"):
        out.append({"text": "Custom weights are used for C, so the ranking depends on them.",
                    "accept": ["weight"]})
    return out


def driver_row(name: str, iata: str, state: str | None, hub: str | None, d: dict) -> dict:
    """One airport's drivers (constraint, growth, size) without RPS: used for comparisons and explanations."""
    g_years = d.get("growth_years")
    return {"airport": f"{name} ({iata})", "iata": iata, "state": state, "faa_hub_size": hub,
            "C": num(d.get("C"), 2),
            "forecast_growth": pct(d.get("growth_share")),
            "forecast_status": growth_status(d.get("growth_source"), d.get("growth_share")),
            "growth_period": f"FY{g_years[0]}–FY{g_years[1]}" if g_years else None,
            "growth_source": d.get("growth_source"),
            "forecast_extra_passengers": people(d.get("extra_passengers")),
            "passengers_last_12_months": people(d.get("passengers_now")),
            "passengers_source": PASSENGER_SOURCE if d.get("passengers_now") is not None else None,
            "signals_measured": signals_measured(d["signals"]),
            "constraint_signals": [_signal(k, d["signals"][k]) for k in SIGNALS],
            "caveats": d.get("caveats", [])}


def drivers_facts(rows: list[dict], question_type: str, period_label: str, horizon: int,
                  scope_notes: list[str]) -> dict:
    """Comparison or explanation from the drivers. RPS is not computed here."""
    flagged = [r["iata"] for r in rows if "⚑" in (r.get("signals_measured") or "")]
    out = {"question_type": question_type, "airports": rows,
           "C_meaning": "0 = least constrained US airport, 1 = most constrained (relative to US airports)",
           "score_formula_not_used": "This answer does not use the RPS score.",
           **common(period_label, horizon, scope_notes, None)}
    if flagged:
        out["flag_note"] = (f"⚑ {', '.join(flagged)}: C is based on fewer than {len(SIGNALS)} signals, because the "
                            "airport is too small to have both busy and quiet hours; compare it with care.")
    return out


def conceptual_facts(question: str, scope_notes: list[str]) -> dict:
    """A general aviation or airport-investment question. No airport data is involved."""
    return {"answer_mode": "conceptual", "question_type": "conceptual", "question": question,
            "scope_notes": scope_notes, "data_period": None, "disclaimer": None}


def research_facts(question: str, airports: list[dict], scope_notes: list[str]) -> dict:
    """A factual question that needs external research. The web notes are added by the web node."""
    return {"answer_mode": "research", "question_type": "research", "question": question,
            "airports": [{"airport": f"{a['name']} ({a['iata']})", "iata": a["iata"], "state": a.get("state")}
                         for a in airports],
            "scope_notes": scope_notes, "data_period": None, "disclaimer": None}


def long_haul_facts(iata: str, name: str, mv, source_note: str, period_label: str, scope_notes: list[str]) -> dict:
    lh = load_config("scoring")["long_haul"]
    d = mv.detail or {}
    return {"question_type": "metric_lookup", "metric": "long-haul share", "airport": f"{name} ({iata})",
            "source": mv.source, "period": mv.period, "flights_counted": d.get("flights"),
            "definitions": {"block_time": f"scheduled block time (gate to gate) of {lh['block_time_minutes'] // 60} hours or more",
                            "distance": f"distance of {lh['distance_miles']:,} miles or more"},
            "long_haul_share_by_block_time": pct(d.get("by_block_time")),
            "long_haul_share_by_distance": pct(d.get("by_distance")),
            "passenger_flights": d.get("passenger_flights"),
            "passenger_long_haul_by_block_time": pct(d.get("passenger_by_block_time")),
            "passenger_long_haul_by_distance": pct(d.get("passenger_by_distance")),
            "international_share_of_flights": pct(d.get("international_share")),
            "flights_without_block_time": d.get("block_time_missing"),
            "caveats": list(mv.caveats) + [source_note],
            **common(period_label, None, scope_notes, None)}


def metric_facts(rows: list[dict], period_label: str, scope_notes: list[str]) -> dict:
    return {"question_type": "metric_lookup", "airports": rows, **common(period_label, None, scope_notes, None)}


def methodology_facts() -> dict:
    cfg, text = load_config("scoring"), load_config("answer_text")["methodology"]
    w = cfg["constraint"]["weights"]
    return {"answer_mode": "methodology", "question_type": "methodology", **text,
            "signals_and_weights": {LABEL[k]: f"{v:.2f}" for k, v in w.items()},
            "horizon_years_default": cfg["horizon"]["default_years"],
            "horizon_years_allowed": cfg["horizon"]["allowed_years"],
            "fallback_growth_per_year": pct(cfg["growth"]["fallback_growth_per_year"], 0),
            "busy_hour": f"at least {cfg['constraint']['busy_hour_min_utilization']:.0%} of the airport's maximum hour",
            "quiet_hour": f"at most {cfg['constraint']['quiet_hour_max_utilization']:.0%} of the airport's maximum hour",
            "maximum_hour": f"the airport's {cfg['constraint']['max_hour_percentile'] * 100:.0f}th-percentile hour",
            "minimum_flights_per_mean": cfg["checks"]["min_flights_per_mean"],
            "disclaimer": load_config("answer_text")["disclaimer"]}
