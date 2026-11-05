"""Scope understanding: whole-country names, flexible region names, code rules on Claude's parse,
ranking by any measure, and plain-language API failures."""

from pathlib import Path

import anthropic
import httpx
import pytest

from airport_agent.graph.nodes import normalize_request, respond
from airport_agent.llm.client import api_problem
from airport_agent.tools.resolve import expand_region, is_national, region_states
from airport_agent.tools.scoring import AirportScore, rank

ROOT = Path(__file__).resolve().parents[2]
needs_data = pytest.mark.skipif(not (ROOT / "data/reference/airports.parquet").exists(), reason="data not built")


@pytest.mark.parametrize("name", ["US", "USA", "U.S.", "U.S.A.", "America", "United States", "nationwide"])
def test_whole_country_names(name):
    assert is_national(name)
    assert len(region_states(name)) >= 50


@pytest.mark.parametrize("name,expected", [("Pacific division", "Pacific"), ("the Pacific region", "Pacific"),
                                           ("New England region", "New England"), ("Midwest states", "Midwest"),
                                           ("South Atlantic division", "South Atlantic")])
def test_generic_words_are_ignored(name, expected):
    assert region_states(name) == region_states(expected) is not None


def test_unofficial_regions_stay_unknown():
    assert region_states("West Coast") is None and region_states("Sun Belt") is None


@needs_data
def test_national_scope_covers_every_airport_with_flights():
    assert len(expand_region("USA")) == len(expand_region("United States")) > 300


def _req(**kw):
    base = dict(intent="rank", airports=[], region=None, metrics=[], follow_up=False, top_n=None,
                rank_by=None, order=None)
    return {**base, **kw}


def test_metric_for_a_whole_scope_becomes_a_ranking_by_that_metric():
    out = normalize_request(_req(intent="metric_lookup", region="US", metrics=["passengers"], top_n=1))
    assert out["intent"] == "rank" and out["rank_by"] == "passengers" and out["top_n"] == 1


def test_metric_for_named_airports_stays_a_lookup():
    out = normalize_request(_req(intent="metric_lookup", airports=["X"], metrics=["passengers"]))
    assert out["intent"] == "metric_lookup"


def test_ranking_with_no_place_covers_the_whole_country():
    assert is_national(normalize_request(_req(rank_by="constraint"))["region"])


def test_follow_up_ranking_keeps_its_previous_scope():
    assert normalize_request(_req(follow_up=True))["region"] is None


def test_top_n_written_in_the_message_is_used_when_claude_left_it_empty():
    assert normalize_request(_req(region="South"), "top 7 fastest growing airports")["top_n"] == 7
    assert normalize_request(_req(region="South", top_n=3), "top 7")["top_n"] == 3   # Claude's value wins


def _score(i, **kw):
    s = AirportScore(i, f"A{i}", "n", None, None, True, None, kw.get("rps", 1.0), None, None, 0.1, "x", None,
                     None, kw.get("pax"), None, kw.get("C", 0.5), None,
                     {k: {"value": None} for k in ("load_factor", "peak_saturation", "taxi_out_excess", "taxi_in_excess")},
                     None, [])
    return s


def test_rank_by_any_measure_and_direction():
    scores = [_score(1, pax=10.0, C=0.2), _score(2, pax=30.0, C=0.9), _score(3, pax=20.0, C=0.5)]
    assert [s.airport_id for s in rank(list(scores), "passengers")] == [2, 3, 1]
    assert [s.airport_id for s in rank(list(scores), "passengers", "lowest")] == [1, 3, 2]
    assert [s.airport_id for s in rank(list(scores), "constraint")][0] == 2


def test_rank_by_a_measure_some_airports_lack():
    scores = [_score(1, pax=None), _score(2, pax=5.0)]
    out = rank(scores, "passengers")
    assert out[0].airport_id == 2 and out[0].rank == 1 and out[1].rank is None


def test_credit_problem_is_named_plainly():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    e = anthropic.BadRequestError("Your credit balance is too low to access the Anthropic API.",
                                  response=httpx.Response(400, request=req), body=None)
    assert "credit balance" in api_problem(e)


def test_api_failure_is_not_blamed_on_the_question():
    state = {"request": None, "messages": [{"role": "user", "content": "q"}],
             "errors": ["understand_request: Claude API unavailable: the account's credit balance is too low"]}
    text = respond(state)["messages"][0]["content"]
    assert "credit balance" in text and "Your question is fine" in text and "rephrase" not in text
