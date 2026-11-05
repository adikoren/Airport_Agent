"""Routing rules (ARCHITECTURE §4.2). Code decides every route by checking facts in the state.

Paths:
- conceptual or methodology: no airport, no data, no search -> explain.
- research (external facts): airports only if an airport must be identified -> web research -> explain.
- supported analysis: resolve -> get data -> metrics -> (score only for rankings) -> facts -> explain.
- follow-ups reuse stored results where everything matches; a new external fact is never reused.
"""

from __future__ import annotations

from airport_agent.graph.nodes import SCORED_INTENTS, _params
from airport_agent.graph.state import AgentState

REUSABLE = {"rank", "compare", "explain_why", "metric_lookup"}


def after_understand(state: AgentState) -> str:
    req = state.get("request")
    if state.get("clarification") or req is None or req["intent"] == "out_of_scope":
        return "respond"
    intent = req["intent"]
    if intent in ("conceptual", "methodology"):
        return "prepare_results"                       # no airport, no data, no search
    if intent == "research":                           # external facts: identify an airport only if one is named
        return "resolve_airports" if (req.get("airports") or req.get("region")) else "prepare_results"
    ctx = state.get("context") or {}
    if req.get("follow_up") and ctx.get("metrics") and intent in REUSABLE and not req.get("metrics"):
        prev = ctx.get("request") or {}
        same_airports = not req.get("airports") or {a.upper() for a in req["airports"]} <= set(ctx.get("airport_codes", []))
        same_region = not req.get("region") or req.get("region") == prev.get("region")
        if same_airports and same_region:
            if intent in SCORED_INTENTS and _params(req, ctx) != ctx.get("params") and ctx.get("scores") is not None:
                return "score_and_rank"                # only scoring parameters changed: re-rank, no fetch
            if intent in SCORED_INTENTS and ctx.get("scores"):
                return "prepare_results"               # same ranking, new presentation
            if intent == "explain_why" and ctx.get("scores"):
                return "prepare_results"               # explain the ranking already computed; nothing is re-fetched
            if intent not in SCORED_INTENTS and prev.get("intent") == intent:
                return "prepare_results"               # same comparison or explanation, new presentation
    return "resolve_airports"


def after_resolve(state: AgentState) -> str:
    if state.get("clarification"):
        return "respond"
    refs = state.get("airport_refs") or []
    intent = state["request"]["intent"]
    if not refs:
        return "prepare_results" if intent == "research" else "respond"
    if any(r["status"] != "resolved" for r in refs):
        return "disambiguate_airports"
    return "prepare_results" if intent == "research" else "get_data"


def after_disambiguate(state: AgentState) -> str:
    if state.get("clarification"):
        return "respond"
    return "prepare_results" if state["request"]["intent"] == "research" else "get_data"


def after_compute(state: AgentState) -> str:
    return "score_and_rank" if state["request"]["intent"] == "rank" else "prepare_results"


def after_prepare(state: AgentState) -> str:
    """Web research when the answer needs it: background for a 'why' question, or external facts."""
    return "web_context" if (state.get("facts") or {}).get("web_mode") else "explain"


def after_guard(state: AgentState) -> str:
    return "respond" if state.get("answer") else "explain"
