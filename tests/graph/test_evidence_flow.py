"""Evidence-first routing: a 'why' question about a ranking already computed is answered from its stored inputs,
with no web search. Mocked Claude; real data (skipped if the data build is absent)."""

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
            "long_haul_threshold": None, "out_of_scope_parts": []}


def req(**kw):
    return ParsedRequest(**{**DEFAULTS, **kw})


class CountingClaude(FakeClaude):
    """Scripted understanding; counts web calls so the test can prove none happened."""

    def __init__(self, requests):
        self.requests = list(requests)
        self.web_calls = 0

    def understand(self, history, ctx):
        return self.requests.pop(0)

    def web_lookup(self, q, codes, gaps):
        self.web_calls += 1
        return []

    def web_context(self, q, codes):
        self.web_calls += 1
        return []


@needs_data
def test_why_about_a_stored_ranking_runs_no_web_search_and_no_data_fetch():
    fake = CountingClaude([req(intent="rank", region="New England"), req(intent="explain_why", follow_up=True)])
    nodes.set_claude(fake)
    graph = build(MemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}
    graph.invoke({"messages": [{"role": "user", "content": "Rank New England airports"}]}, config=cfg)
    ran = []
    for update in graph.stream({"messages": [{"role": "user", "content": "Why is the first one above the second?"}]},
                               config=cfg, stream_mode="updates"):
        ran += list(update)
    nodes.set_claude(None)
    assert fake.web_calls == 0
    assert "web_context" not in ran and "get_data" not in ran and "score_and_rank" not in ran


def _turn(graph, cfg, text):
    ran = []
    for update in graph.stream({"messages": [{"role": "user", "content": text}]}, config=cfg, stream_mode="updates"):
        ran += list(update)
    return ran, graph.get_state(cfg).values


@needs_data
def test_rank_then_why_then_horizon_keeps_the_ranking_population_and_display_count():
    """Rank New England → why is the first above the second → 10-year horizon: the last turn ranks everyone again."""
    first_req = req(intent="rank", region="New England")
    fake = CountingClaude([first_req,
                           req(intent="explain_why", follow_up=True, airports=["BOS", "BDL"]),
                           req(intent="rank", follow_up=True, horizon_years=10)])
    nodes.set_claude(fake)
    graph = build(MemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}
    try:
        _, s1 = _turn(graph, cfg, "Rank New England airports")
        top_two = [r["iata"] for r in s1["facts"]["ranking"][:2]]
        scope1 = (s1["facts"]["airports_in_scope"], s1["facts"]["airports_shown"], len(s1["facts"]["ranking"]))

        ran2, s2 = _turn(graph, cfg, f"Why is {top_two[0]} above {top_two[1]}?")
        assert "get_data" not in ran2 and "score_and_rank" not in ran2 and fake.web_calls == 0
        assert s2["request"]["top_n"] is None                       # discussing two airports sets no count

        ran3, s3 = _turn(graph, cfg, "Use a 10-year horizon instead.")
        assert "score_and_rank" in ran3 and "get_data" not in ran3   # re-scored, not re-fetched
        f3 = s3["facts"]
        assert f3["horizon_years"] == 10
        assert (f3["airports_in_scope"], f3["airports_shown"], len(f3["ranking"])) == scope1
        assert f3["scope"] == s1["facts"]["scope"]
        assert {r["iata"] for r in f3["ranking"]} >= set(top_two)   # the discussed airports are still in the list
    finally:
        nodes.set_claude(None)


@needs_data
def test_why_turn_without_follow_up_flag_does_not_narrow_the_next_ranking():
    """If the model misses the follow-up flag, the 'why' turn re-resolves two airports. The ranking population that
    the next horizon follow-up uses must still be the full earlier population."""
    fake = CountingClaude([req(intent="rank", region="New England"),
                           req(intent="explain_why", follow_up=False, airports=["BOS", "BDL"]),
                           req(intent="rank", follow_up=True, horizon_years=10)])
    nodes.set_claude(fake)
    graph = build(MemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}
    try:
        _, s1 = _turn(graph, cfg, "Rank New England airports")
        size1 = s1["facts"]["airports_in_scope"]
        _turn(graph, cfg, "Why is BOS above BDL?")
        _, s3 = _turn(graph, cfg, "Use a 10-year horizon instead.")
        assert s3["facts"]["airports_in_scope"] == size1
        assert s3["facts"]["scope"] == s1["facts"]["scope"]
    finally:
        nodes.set_claude(None)
