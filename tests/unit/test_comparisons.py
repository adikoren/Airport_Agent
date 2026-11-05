"""Comparative and superlative claims are checked against the figures; research fallbacks keep their target."""

from airport_agent.graph.nodes import resolve_ranked_positions
from airport_agent.tools import output_guard as G

ROWS = [
    {"iata": "BTV", "airport": "Burlington (BTV)", "rank": 2, "forecast_growth": "12.0%", "RPS": "7.6", "C": "0.64"},
    {"iata": "ORH", "airport": "Worcester (ORH)", "rank": 4, "forecast_growth": "10.8%", "RPS": "6.0", "C": "0.55"},
    {"iata": "MVY", "airport": "Martha's Vineyard (MVY)", "rank": 8, "forecast_growth": "32.4%", "RPS": "4.1",
     "C": "0.13"},
    {"iata": "BOS", "airport": "Boston Logan (BOS)", "rank": 1, "forecast_growth": "13.8%", "RPS": "8.1", "C": "0.59"},
]


def computed(**extra):
    return {"answer_mode": "computed", "ranking": ROWS, "follow_up": True,
            "essential_caveats": [], "disclaimer": "x", **extra}


# ---- comparisons against the figures

def test_a_comparison_contradicted_by_the_figures_is_reported():
    problems = G.comparison_problems("BTV's forecast growth (12.0%) is below ORH's (10.8%).", ROWS)
    assert problems and "does not match the figures" in problems[0]


def test_a_comparison_that_matches_the_figures_passes():
    assert G.comparison_problems("BTV's forecast growth (12.0%) is above ORH's (10.8%).", ROWS) == []
    assert G.comparison_problems("ORH's forecast growth is lower than BTV's.", ROWS) == []


def test_a_claim_about_rank_position_alone_is_not_checked_as_a_figure():
    assert G.comparison_problems("BOS ranks above BTV, the second airport.", ROWS) == []


def test_a_sentence_with_two_directions_is_left_alone():
    assert G.comparison_problems("BTV's growth is higher than ORH's, although BTV's C is lower.", ROWS) == []


# ---- superlatives against the figures

def test_highest_growth_claim_for_a_non_maximum_airport_is_reported():
    problems = G.comparison_problems("BOS leads because it has the highest forecast growth among the ranked airports.",
                                     ROWS)
    assert problems and "superlative" in problems[0]


def test_highest_growth_claim_for_the_maximum_airport_passes():
    assert G.comparison_problems("MVY has the highest forecast growth among the ranked airports.", ROWS) == []


def test_check_reports_a_wrong_comparison_in_a_computed_answer():
    problems = G.check("### Why\nBOS leads because it has the highest forecast growth.\n\n"
                       "This is a screening signal, not investment advice.\n### Assumptions & uncertainty\nx",
                       computed(follow_up=False))
    assert any("superlative" in p for p in problems)


# ---- research: retrieved figures, and the fallback keeps its target

def test_a_research_figure_from_the_retrieved_text_passes_without_a_web_heading():
    facts = {"answer_mode": "research", "airports": [{"airport": "Boston Logan (BOS)", "iata": "BOS"}],
             "web": {"status": "results", "notes": [{"text": "The terminal opens in 2029.",
                                                     "sources": [{"url": "https://airport.example/plans",
                                                                  "title": "Plans", "date": "2026"}]}]}}
    draft = "### Projects\nThe terminal opens in 2029 ([Plans](https://airport.example/plans))."
    assert G.check(draft, facts) == []


def test_research_fallback_names_its_airport_and_has_no_investment_disclaimer():
    facts = {"answer_mode": "research", "airports": [{"airport": "Boston Logan (BOS)", "iata": "BOS"}],
             "disclaimer": "This is a screening signal, not investment advice.",
             "web": {"status": "no_reliable_result", "notes": []}}
    text = G.template(facts, "what is planned at the top airport?")
    assert "Boston Logan (BOS)" in text
    assert "investment advice" not in text


# ---- ranked positions resolve in code

def _ctx():
    return {"scores": [{"iata": "BOS", "rank": 1}, {"iata": "BTV", "rank": 2}, {"iata": "PWM", "rank": 3},
                       {"iata": "HYA", "rank": None}]}


def test_top_ranked_position_resolves_to_the_first_stored_rank():
    req, clar = resolve_ranked_positions({"intent": "research", "airports": [], "ranked_positions": [1]}, _ctx())
    assert clar is None and req["airports"] == ["BOS"] and req["resolved_from_ranking"] == ["BOS"]


def test_a_generic_mention_is_dropped_so_it_cannot_reach_disambiguation():
    req, _ = resolve_ranked_positions({"intent": "research", "airports": ["top-ranked airport"],
                                       "ranked_positions": [1]}, _ctx())
    assert req["airports"] == ["BOS"]


def test_a_real_airport_named_with_a_position_is_kept_too():
    req, _ = resolve_ranked_positions({"intent": "explain_why", "airports": ["Boston"], "ranked_positions": [2]}, _ctx())
    assert req["airports"] == ["BTV", "Boston"]


def test_positions_resolve_in_rank_order_not_list_order():
    req, _ = resolve_ranked_positions({"intent": "explain_why", "airports": [], "ranked_positions": [2, 1]}, _ctx())
    assert req["airports"] == ["BTV", "BOS"]


def test_no_stored_ranking_gives_a_clarification():
    _, clar = resolve_ranked_positions({"intent": "research", "airports": [], "ranked_positions": [1]}, {})
    assert clar and "don't have a ranking" in clar["text"]


def test_a_ranking_request_is_not_changed_by_positions():
    req, clar = resolve_ranked_positions({"intent": "rank", "airports": [], "ranked_positions": [1]}, _ctx())
    assert clar is None and req["airports"] == []

