"""Regression: a research question about 'the top-ranked airport' must research the airport the stored ranking gives,
not another one, and the explanation of that ranking must not search the web. Mocked Claude; real data."""

import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from airport_agent.graph import nodes
from airport_agent.graph.builder import build
from airport_agent.llm.fake import FakeClaude
from airport_agent.llm.schemas import ParsedRequest

ROOT = Path(__file__).resolve().parents[2]
needs_data = pytest.mark.skipif(not (ROOT / "data/reference/signal_reference.parquet").exists(),
                                reason="data not built")
DEFAULTS = {"follow_up": False, "top_n": None, "airports": [], "metrics": [], "region": None, "rank_by": None,
            "order": None, "horizon_years": None, "weights": None, "long_haul_definition": None,
            "long_haul_threshold": None, "out_of_scope_parts": [], "ranked_positions": []}


def req(**kw):
    return ParsedRequest(**{**DEFAULTS, **kw})


class Recording(FakeClaude):
    """Scripted understanding; records every web query so the test can see what was researched."""

    def __init__(self, requests):
        self.requests = list(requests)
        self.web_queries = []

    def understand(self, history, ctx):
        return self.requests.pop(0)

    def web_lookup(self, question, codes, gaps):
        self.web_queries.append(("lookup", question, list(codes)))
        return [{"text": "- Terminal work is planned [Airport site, 2026]",
                 "sources": [{"url": "https://airport.example/plans", "title": "Airport plans", "date": "2026"}]}]

    def web_context(self, question, codes):
        self.web_queries.append(("context", question, list(codes)))
        return []


def run_turns(fake, texts):
    nodes.set_claude(fake)
    graph = build(MemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}
    out = []
    try:
        for text in texts:
            ran = []
            for update in graph.stream({"messages": [{"role": "user", "content": text}]}, config=cfg,
                                       stream_mode="updates"):
                ran += list(update)
            out.append((ran, graph.get_state(cfg).values))
    finally:
        nodes.set_claude(None)
    return out


@needs_data
def test_top_ranked_airport_is_researched_not_another_airport():
    """The regression: the research must target the first airport of the stored ranking (BOS in the report)."""
    fake = Recording([req(intent="rank", region="New England"),
                      req(intent="research", follow_up=True, ranked_positions=[1])])
    (_, s1), (_, s2) = run_turns(fake, ["Rank New England airports",
                                           "What expansion projects are currently planned at the top-ranked airport?"])
    top = s1["facts"]["ranking"][0]["iata"]
    assert s2["request"]["resolved_from_ranking"] == [top]
    assert fake.web_queries == [("lookup", fake.web_queries[0][1], [top])]     # researched exactly that airport
    assert [a["iata"] for a in s2["facts"]["airports"]] == [top]
    answer = s2["messages"][-1]["content"]
    assert top in answer
    for other in ("BDL", "BTV", "PWM"):
        if other != top:
            assert other not in answer                                        # no other airport appears


@needs_data
def test_generic_airport_wording_never_reaches_disambiguation():
    """Parser gives the phrase as a mention and the position: no disambiguation, so no model choice of an airport."""
    fake = Recording([req(intent="rank", region="New England"),
                      req(intent="research", follow_up=True, airports=["top-ranked airport"], ranked_positions=[1])])
    (_, s1), (ran2, _) = run_turns(fake, ["Rank New England airports", "What is planned at the top-ranked airport?"])
    top = s1["facts"]["ranking"][0]["iata"]
    assert "disambiguate_airports" not in ran2
    assert fake.web_queries[0][2] == [top]


@needs_data
def test_second_ranked_airport_is_resolved_from_the_position():
    fake = Recording([req(intent="rank", region="New England"),
                      req(intent="research", follow_up=True, ranked_positions=[2])])
    (_, s1), _ = run_turns(fake, ["Rank New England airports", "What are the plans at the second one?"])
    second = s1["facts"]["ranking"][1]["iata"]
    assert fake.web_queries[0][2] == [second]


@needs_data
def test_reference_without_a_stored_ranking_asks_and_searches_nothing():
    fake = Recording([req(intent="research", ranked_positions=[1])])
    ((ran, s),) = run_turns(fake, ["What is planned at the top-ranked airport?"])
    assert fake.web_queries == []
    assert "web_context" not in ran
    assert "don't have a ranking" in s["messages"][-1]["content"]


@needs_data
def test_position_outside_the_ranking_asks_the_user():
    fake = Recording([req(intent="rank", region="New England"),
                      req(intent="research", follow_up=True, ranked_positions=[99])])
    (_, _), (ran2, s2) = run_turns(fake, ["Rank New England airports", "What is planned at airport 99?"])
    assert fake.web_queries == [] and "web_context" not in ran2
    assert "is not in it" in s2["messages"][-1]["content"]


@needs_data
def test_full_sequence_rank_then_research_then_score_explanation():
    """Rank → research the top-ranked airport → explain its score. The explanation runs no data step and no search."""
    fake = Recording([req(intent="rank", region="New England"),
                      req(intent="research", follow_up=True, ranked_positions=[1]),
                      req(intent="explain_why", follow_up=True, airports=[], ranked_positions=[1, 2])])
    (_, s1), _, (ran3, s3) = run_turns(fake, ["Rank New England airports",
                                                    "What expansion projects are currently planned at the top-ranked airport?",
                                                    "Explain the score: why is the first above the second?"])
    top, second = s1["facts"]["ranking"][0]["iata"], s1["facts"]["ranking"][1]["iata"]
    assert len(fake.web_queries) == 1                                          # only the research turn searched
    assert "web_context" not in ran3 and "get_data" not in ran3 and "score_and_rank" not in ran3
    assert s3["request"]["resolved_from_ranking"] == [top, second]
    assert s3["facts"]["answer_mode"] == "computed"
