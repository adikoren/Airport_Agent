"""Evidence rules: metric labels match the calculation, coverage and forecast status are stated, ranges say what they
measure, and equal rounded values are not treated as a problem. No Claude, no data files: plain inputs."""

import json

from airport_agent.graph import nodes
from airport_agent.tools import facts as F
from airport_agent.tools import output_guard as G
from airport_agent.tools.scoring import SIGNALS, AirportScore


def score(iata="ABC", rps=10.0, growth=0.2, source="FAA TAF", C=0.5, rps_range=None, rank=1, scored=True):
    sig = {k: {"value": 0.5, "percentile": 0.5, "missing_reason": None} for k in SIGNALS}
    lo, hi = rps_range if rps_range else (None, None)
    return AirportScore(airport_id=1, iata=iata, name=f"{iata} Test Airport", state="MA", hub_size="small",
                        scored=scored, not_scored_reason=None if scored else "load factor missing",
                        rps=rps, rps_low=lo, rps_high=hi, growth_share=growth, growth_source=source,
                        growth_years=(2026, 2031), taf_edition="2025", passengers_now=2_000_000.0,
                        extra_passengers=400_000.0, C=C, C_range=None, signals=sig, international_share=0.1,
                        caveats=[], rank=rank)


# ---- labels match what is calculated

def test_taxi_labels_describe_the_busy_minus_quiet_calculation_not_a_cause():
    assert "busy-hour minus quiet-hour" in F.LABEL["taxi_in_excess"]
    assert "busy-hour minus quiet-hour" in F.LABEL["taxi_out_excess"]
    assert "waiting for a gate" not in F.LABEL["taxi_in_excess"]
    assert "runway queue" not in F.LABEL["taxi_out_excess"]


def test_measure_names_in_rankings_use_the_same_labels():
    assert F.MEASURES["taxi_in_excess"] == F.LABEL["taxi_in_excess"]
    assert F.MEASURES["load_factor"] == F.LABEL["load_factor"]


# ---- passenger coverage

def test_passenger_figures_state_their_source_and_coverage():
    row = F.airport_row(score())
    assert row["passengers_source"] == F.PASSENGER_SOURCE
    assert "boarding" in F.PASSENGER_SOURCE and "domestic and international" in F.PASSENGER_SOURCE


def test_no_passenger_source_when_there_is_no_passenger_figure():
    s = score()
    s.passengers_now = None
    assert F.airport_row(s)["passengers_source"] is None


# ---- missing forecast vs non-positive growth

def test_missing_faa_forecast_is_stated_as_fallback_not_as_no_growth():
    status = F.growth_status("fallback", 0.0824)
    assert "no FAA forecast" in status and "fallback" in status and "not an FAA figure" in status
    assert "no growth" not in status


def test_non_positive_faa_growth_is_stated_as_no_growth():
    assert "no growth over the horizon" in F.growth_status("FAA TAF", 0.0)
    assert "not positive" in F.growth_status("FAA TAF", -0.02)


def test_positive_faa_growth_and_unknown_growth():
    assert F.growth_status("FAA TAF", 0.15) == "FAA forecast"
    assert F.growth_status("FAA TAF", None) is None


def test_zero_rps_note_says_whether_growth_or_constraint_is_the_reason():
    no_growth = F.airport_row(score(rps=0.0, growth=0.0))
    assert no_growth["note"] == "FAA forecast shows no growth over the horizon, so RPS is 0"
    no_constraint = F.airport_row(score(rps=0.0, growth=0.2, C=0.0))
    assert no_constraint["note"] == "constraint C is 0, so RPS is 0"
    assert "no demand growth forecast" not in json.dumps([no_growth, no_constraint])


# ---- ranges say what they measure

def test_ranking_with_rps_range_says_the_range_is_constraint_only():
    f = F.ranking_facts([score(rps_range=(9.0, 11.0))], "New England", "period", 5, [], {})
    assert f["range_meaning"] == F.RANGE_MEANING
    assert "constraint C" in f["range_meaning"] and "growth range is not measured" in f["range_meaning"]


def test_ranking_without_range_has_no_range_statement():
    f = F.ranking_facts([score()], "New England", "period", 5, [], {})
    assert "range_meaning" not in f


def test_template_labels_the_range_as_constraint_only():
    f = F.ranking_facts([score(rps_range=(9.0, 11.0))], "New England", "period", 5, [], {})
    text = G.template(f, "rank New England airports")
    assert "RPS range (constraint only)" in text and F.RANGE_MEANING in text


# ---- equal rounded values are not a warning

def test_equal_rounded_values_do_not_produce_a_tie_warning():
    a = score(iata="AAA", rps=8.04, rank=1)
    b = score(iata="BBB", rps=8.01, rank=2)            # both show as 8.0 at one decimal
    f = F.ranking_facts([a, b], "New England", "period", 5, [], {})
    text = json.dumps(f).lower()
    assert "flag_note" not in f
    assert "tie" not in text and "separated" not in text and "overlap" not in text


# ---- follow-ups keep the earlier scope

def test_horizon_follow_up_keeps_the_earlier_scope():
    ctx = {"params": {"horizon": 5, "rank_by": "constraint", "order": "lowest", "top_n": 3, "weights": None}}
    params = nodes._params({"horizon_years": 10, "follow_up": True}, ctx)
    assert params["horizon"] == 10
    assert params["rank_by"] == "constraint" and params["order"] == "lowest" and params["top_n"] == 3


def test_a_new_question_does_not_inherit_scope():
    ctx = {"params": {"horizon": 10, "rank_by": "constraint", "order": "lowest", "top_n": 3}}
    params = nodes._params({"follow_up": False}, ctx)
    assert params == {"horizon": 5, "weights": None, "rank_by": "rps", "order": "highest", "top_n": None}


# ---- required sections depend on the answer context

def _facts(follow_up, rows=None, horizon=5, **kw):
    f = F.ranking_facts(rows or [score(iata="AAA", source="fallback", growth=0.0824)], "New England", "period",
                        horizon, [], {})
    f.update(follow_up=follow_up, essential_caveats=F.essential_caveats(f), **kw)
    return f


def test_initial_answer_must_carry_the_section_and_the_screening_sentence():
    problems = G.check("The top airport is AAA Test Airport.", _facts(follow_up=False))
    assert any("Assumptions & uncertainty" in p for p in problems)
    assert any("not investment advice" in p for p in problems)


def test_focused_follow_up_need_not_repeat_the_section_or_the_sentence():
    draft = "AAA ranks first; its growth figure is a fallback rate, not an FAA figure."
    assert G.check(draft, _facts(follow_up=True)) == []


def test_follow_up_still_states_an_essential_caveat():
    problems = G.check("AAA ranks first.", _facts(follow_up=True))
    assert problems == [f"missing a caveat needed for this answer: {F.essential_caveats(_facts(True))[0]['text']}"]


def test_zero_rps_reason_is_essential_and_stated_in_the_template():
    f = _facts(follow_up=True, rows=[score(iata="ZZZ", rps=0.0, growth=0.0, source="FAA TAF")])
    (zero,) = [c for c in f["essential_caveats"] if "RPS 0" in c["accept"][0] or "RPS 0" in c["accept"]]
    assert zero["text"].startswith("Airports with RPS 0")
    text = G.template(f, "why is ZZZ ranked last")
    assert zero["text"] in text and G.check(text, f) == []


def test_changed_horizon_is_an_essential_caveat_for_a_follow_up():
    f = _facts(follow_up=True, rows=[score(iata="AAA", source="FAA TAF", growth=0.2)], horizon=10)
    assert any(c["accept"] == ["10 year", "10-year", "horizon"] for c in f["essential_caveats"])
    assert G.check("AAA ranks first over a 10-year horizon.", f) == []
    assert G.check("AAA ranks first.", f) != []


def test_follow_up_template_has_no_section_or_sentence_but_keeps_the_caveats():
    f = _facts(follow_up=True)
    text = G.template(f, "why is AAA first")
    assert "Assumptions & uncertainty" not in text and "not investment advice" not in text
    assert "fallback" in text
    assert G.check(text, f) == []


def test_initial_template_has_the_section_and_passes_the_check():
    f = _facts(follow_up=False)
    text = G.template(f, "rank New England airports")
    assert "Assumptions & uncertainty" in text
    assert G.check(text, f) == []


def test_zero_rps_airports_are_listed_not_ranked_and_ranked_airports_outside_the_count_are_reported():
    from airport_agent.tools.scoring import rank
    scores = [score(iata=f"R{i:02d}", rps=100.0 - i, growth=0.2, rank=None) for i in range(25)]
    scores.append(score(iata="ZER", rps=0.0, growth=0.0, rank=None))
    ranked = rank(scores, "rps", "highest")
    f = F.ranking_facts(ranked, "All", "period", 5, [], {})          # 25 ranked: a large scope shows the top 10
    assert f["airports_ranked"] == 25 and f["airports_shown"] == 10 and len(f["ranking"]) == 10
    assert [r["iata"] for r in f["ranking"]] == [f"R{i:02d}" for i in range(10)]
    assert [r["rank"] for r in f["ranked_not_shown"]] == list(range(11, 21))   # ranked, just not displayed
    assert [r["iata"] for r in f["listed_not_ranked"]] == ["ZER"]           # RPS 0: no rank, with its reason
    assert "RPS is 0" in f["listed_not_ranked"][0]["note"]


def test_explain_prompt_forbids_suggesting_a_data_error_from_equal_rounded_values():
    from airport_agent.llm.client import prompt
    text = prompt("explain").lower()
    assert "never suggest a data error" in text and "check this" in text
    assert "rounded value" in text
