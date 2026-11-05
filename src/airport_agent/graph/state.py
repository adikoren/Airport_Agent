"""Agent state (ARCHITECTURE §4.3). Each node writes only its own keys."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]   # chat history: {"role": "user"|"assistant", "content": str}
    request: dict | None                            # ParsedRequest as dict, plus "airports_from_context"
    airport_refs: list[dict]                        # AirportRef as dict
    clarification: dict | None                      # {"text": str} question for the user
    raw_data: dict                                  # per source: what was fetched (OAG long haul, data coverage)
    data_quality: dict                              # per airport: ok / missing with reasons
    metrics: dict                                   # per airport iata: {metric: MetricValue dict}
    scores: list[dict]                              # AirportScore dicts, ranked
    facts: dict                                     # the only numbers Claude may quote
    web_context: list[dict]                         # web notes: text, sources (url, title, date), kind
    web_status: str | None                          # results | no_reliable_result | failed
    web_error: str | None
    draft_answer: str | None
    answer: str | None
    guard_problems: list[str]
    context: dict                                   # saved for follow-ups
    counters: dict                                  # claude_calls, api_calls, explain_retries
    errors: list[str]
    route: str | None                               # set by the routing helpers for tracing


NODE_WRITES: dict[str, set[str]] = {
    "understand_request": {"request", "counters", "errors", "route", "airport_refs", "clarification", "facts",
                           "scores", "web_context", "web_status", "web_error", "draft_answer", "answer", "guard_problems", "raw_data",
                           "data_quality", "metrics"},
    "resolve_airports": {"airport_refs", "clarification", "errors"},
    "disambiguate_airports": {"airport_refs", "clarification", "counters", "errors"},
    "get_data": {"raw_data", "data_quality", "counters", "errors"},
    "compute_metrics": {"metrics", "errors"},
    "score_and_rank": {"scores", "errors"},
    "prepare_results": {"facts", "errors"},
    "web_context": {"web_context", "web_status", "web_error", "counters", "errors"},
    "explain": {"draft_answer", "counters", "errors"},
    "output_guard": {"answer", "guard_problems", "counters"},
    "respond": {"messages", "context"},
}


def empty_counters() -> dict[str, Any]:
    return {"claude_calls": 0, "api_calls": 0, "explain_retries": 0}
