"""Answer modes and routing (spec §7). Mocked model and web; real data. Phrasings are not the prompt's examples."""

import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from airport_agent.graph import nodes
from airport_agent.graph.builder import build
from airport_agent.llm.client import ClaudeUnavailable
from airport_agent.llm.fake import FakeClaude
from airport_agent.llm.schemas import ParsedRequest
from airport_agent.tools import output_guard as G

ROOT = Path(__file__).resolve().parents[2]
needs_data = pytest.mark.skipif(not (ROOT / "data/reference/signal_reference.parquet").exists(), reason="data not built")
SEARCH_FREE = {"get_data", "compute_metrics", "score_and_rank", "web_context", "resolve_airports"}


def req(**kw):
    return ParsedRequest(**{**dict(follow_up=False, top_n=None, airports=[], metrics=[], region=None, rank_by=None,
                                   order=None, horizon_years=None, weights=None, long_haul_definition=None,
                                   long_haul_threshold=None, out_of_scope_parts=[]), **kw})


class Scripted(FakeClaude):
    def __init__(self, requests, web_notes=None, web_error=None):
        self.requests = list(requests)
        self.web_notes = web_notes
        self.web_error = web_error
        self.web_calls = 0

    def understand(self, history, ctx):
        return self.requests.pop(0)

    def _web(self):
        self.web_calls += 1
        if self.web_error:
            raise ClaudeUnavailable(self.web_error)
        return self.web_notes if self.web_notes is not None else []

    def web_lookup(self, q, codes, gaps):
        return self._web()

    def web_context(self, q, codes):
        return self._web()


def turn(fake, text, thread=None):
    """Run one turn; return (nodes that ran, final state)."""
    nodes.set_claude(fake)
    graph = build(MemorySaver())
    cfg = {"configurable": {"thread_id": thread or str(uuid.uuid4())}, "recursion_limit": 25}
    ran = []
    for update in graph.stream({"messages": [{"role": "user", "content": text}]}, config=cfg, stream_mode="updates"):
        ran += list(update)
    state = graph.get_state(cfg).values
    nodes.set_claude(None)
    return ran, state, cfg


def sourced(text="- It has 115 gates [airport site, 2024]"):
    return [{"text": text, "sources": [{"url": "https://airport.example/facts", "title": "Airport facts", "date": "2024"}]}]


# ---------------------------------------------------------------- routing by answer mode

def test_conceptual_question_needs_no_airport_data_or_search():
    fake = Scripted([req(intent="conceptual")])
    ran, state, _ = turn(fake, "What is load factor?")
    assert not (set(ran) & SEARCH_FREE)
    assert state["facts"]["answer_mode"] == "conceptual"
    assert fake.web_calls == 0


def test_conceptual_why_question_is_not_scored():
    fake = Scripted([req(intent="conceptual")])
    ran, state, _ = turn(fake, "Why do planners care about peak hours?")
    assert "score_and_rank" not in ran and "get_data" not in ran


@needs_data
def test_current_airport_fact_goes_straight_to_research():
    fake = Scripted([req(intent="research", airports=["San Francisco International"])], web_notes=sourced())
    ran, state, _ = turn(fake, "How many gates does this airport have?")
    assert "web_context" in ran and fake.web_calls == 1
    assert not (set(ran) & {"get_data", "compute_metrics", "score_and_rank"})


def test_published_budget_is_research_not_refused():
    fake = Scripted([req(intent="research", airports=["ANC"])],
                    web_notes=sourced("- The published budget is $1.2 billion [city council, 2025]"))
    ran, state, _ = turn(fake, "What did the published budget for the terminal project say?")
    assert "web_context" in ran
    assert "I can't do that part" not in state["messages"][-1]["content"]


def test_unsupported_irr_projection_gets_no_invented_result():
    fake = Scripted([req(intent="out_of_scope")])
    ran, state, _ = turn(fake, "What IRR will a new terminal earn over ten years?")
    text = state["messages"][-1]["content"]
    assert "I can't do that part" in text and "recommend" in text and "return" in text
    assert "web_context" not in ran and "get_data" not in ran


@needs_data
def test_supported_congestion_comparison_is_computed_without_a_score():
    fake = Scripted([req(intent="compare", airports=["SFO", "LAX"])])
    ran, state, _ = turn(fake, "Which of two big West Coast hubs is more constrained?")
    assert "compute_metrics" in ran and "score_and_rank" not in ran
    assert state["facts"]["answer_mode"] == "computed"
    assert "score_formula_not_used" in state["facts"]


@needs_data
def test_why_question_does_not_compute_rps():
    fake = Scripted([req(intent="explain_why", airports=["SFO"])], web_notes=sourced("- Runways are shared [FAA, 2024]"))
    ran, state, _ = turn(fake, "What holds SFO back today?")
    assert "score_and_rank" not in ran and "score_formula_not_used" in state["facts"]


@needs_data
def test_new_external_fact_after_results_is_not_reused_from_them():
    fake = Scripted([req(intent="rank", region="New England"), req(intent="research", follow_up=True, airports=["ANC"])],
                    web_notes=sourced())
    nodes.set_claude(fake)
    graph = build(MemorySaver())
    c = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}
    graph.invoke({"messages": [{"role": "user", "content": "Rank New England airports"}]}, config=c)
    ran = []
    for u in graph.stream({"messages": [{"role": "user", "content": "Who owns this airport?"}]}, config=c,
                          stream_mode="updates"):
        ran += list(u)
    nodes.set_claude(None)
    assert "web_context" in ran and fake.web_calls == 1


@needs_data
def test_follow_up_that_asks_to_explain_results_reuses_them():
    fake = Scripted([req(intent="rank", region="New England"), req(intent="explain_why", follow_up=True)])
    graph = build(MemorySaver())
    c = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}
    nodes.set_claude(fake)
    for text in ["Rank New England airports", "Why is the first one first?"]:
        ran = [k for u in graph.stream({"messages": [{"role": "user", "content": text}]}, config=c, stream_mode="updates") for k in u]
    nodes.set_claude(None)
    assert "get_data" not in ran and "score_and_rank" not in ran


# ---------------------------------------------------------------- failures and provenance

def test_failed_search_is_reported_as_failed_not_as_nothing_found():
    fake = Scripted([req(intent="research", airports=["ANC"])],
                    web_error="LLM API unavailable: the Anthropic account's credit balance is too low")
    ran, state, _ = turn(fake, "Who owns the ANC airport?")
    text = state["messages"][-1]["content"]
    assert state["web_status"] == "failed"
    assert "did not run" in text and "credit balance" in text
    assert "searched the web" not in text.lower() and "found nothing" not in text.lower()


def test_search_that_ran_with_no_reliable_result_says_so():
    fake = Scripted([req(intent="research", airports=["ANC"])], web_notes=[])
    _, state, _ = turn(fake, "Who owns the ANC airport?")
    assert state["web_status"] == "no_reliable_result"
    assert "found no reliable source" in state["messages"][-1]["content"]


def test_research_answer_keeps_source_title_and_link():
    fake = Scripted([req(intent="research", airports=["ANC"])], web_notes=sourced())
    _, state, _ = turn(fake, "Who owns the ANC airport?")
    text = state["messages"][-1]["content"]
    assert "[Airport facts](https://airport.example/facts), 2024" in text


def test_metro_name_asks_for_one_or_all_instead_of_taking_the_largest():
    fake = Scripted([req(intent="compare", airports=["Chicago"])])
    ran, state, _ = turn(fake, "How busy are the Chicago airports?")
    text = state["messages"][-1]["content"]
    assert "ORD" in text and "MDW" in text and "explain" not in ran


# ---------------------------------------------------------------- the number check, per mode

def test_conceptual_definition_with_numbers_passes_but_airport_statistic_is_rejected():
    facts = {"answer_mode": "conceptual", "question": "What is load factor?"}
    ok = "Load factor is passengers divided by seats, for example 80 of 100 seats is 80%."
    bad = "Boston (BOS) has a load factor of 83.4% this year."
    assert G.check(ok, facts, "What is load factor?") == []
    assert G.check(bad, facts, "What is load factor?")


def test_labelled_illustration_is_allowed_in_a_conceptual_answer():
    facts = {"answer_mode": "conceptual", "question": "What is load factor?"}
    assert G.check("Illustration: an invented Airport A with 90 of 100 seats gives 90%.", facts, "") == []
    # a real airport code in an example, not named in the question, is still rejected
    assert G.check("Illustration: BOS with 90 of 100 seats gives 90%.", facts, "")


def test_sourced_figure_without_a_retrieved_source_is_rejected():
    facts = {"answer_mode": "research", "web": {"status": "results", "notes": sourced()}}
    draft = "### From web sources (not from the agent's data)\n- It has 115 gates [airport site]"
    assert any("source reference" in p for p in G.check(draft, facts, ""))


def test_invented_citation_or_figure_is_rejected():
    facts = {"answer_mode": "research", "web": {"status": "results", "notes": sourced()}}
    invented_url = ("### From web sources (not from the agent's data)\n- It has 115 gates "
                    "[Source](https://made-up.example/gates)")
    invented_figure = ("### From web sources (not from the agent's data)\n- It has 999 gates "
                       "[Airport facts](https://airport.example/facts)")
    assert G.check(invented_url, facts, "")
    assert any("999" in p for p in G.check(invented_figure, facts, ""))


def test_claiming_a_search_that_did_not_run_is_rejected():
    facts = {"answer_mode": "research", "web": {"status": "failed", "error": "down", "notes": []}}
    assert any("web search" in p for p in G.check("I searched the web and found nothing.", facts, ""))


def test_computed_number_on_the_wrong_airport_is_rejected():
    facts = {"answer_mode": "computed", "question_type": "rank", "ranked_by": "x", "horizon_years": 5,
             "ranking": [{"airport": "Boston (BOS)", "iata": "BOS", "RPS": "8.1"},
                         {"airport": "Portland (PWM)", "iata": "PWM", "RPS": "6.1"}],
             "assumptions": [], "data_period": "p", "scope_notes": [], "disclaimer": "This is a screening signal, not investment advice."}
    body = "\n\n### Assumptions & uncertainty\n- x\n\nThis is a screening signal, not investment advice."
    assert G.check("Boston (BOS) has RPS 8.1." + body, facts, "") == []
    assert G.check("Boston (BOS) has RPS 6.1." + body, facts, "")
