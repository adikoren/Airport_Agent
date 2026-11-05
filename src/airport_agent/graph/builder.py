"""Build the LangGraph graph: 11 nodes and the routing of ARCHITECTURE §4.1–4.2."""

from __future__ import annotations

import sqlite3

from langgraph.graph import END, START, StateGraph

from airport_agent.graph import nodes as n
from airport_agent.graph import routing as r
from airport_agent.graph.state import AgentState
from airport_agent.settings import get_settings


def build(checkpointer=None):
    g = StateGraph(AgentState)
    for name in ["understand_request", "resolve_airports", "disambiguate_airports", "get_data", "compute_metrics",
                 "score_and_rank", "prepare_results", "web_context", "explain", "output_guard", "respond"]:
        g.add_node(name, getattr(n, name))
    g.add_edge(START, "understand_request")
    g.add_conditional_edges("understand_request", r.after_understand,
                            ["respond", "prepare_results", "score_and_rank", "resolve_airports"])
    g.add_conditional_edges("resolve_airports", r.after_resolve,
                            ["respond", "disambiguate_airports", "get_data", "prepare_results"])
    g.add_conditional_edges("disambiguate_airports", r.after_disambiguate, ["respond", "get_data", "prepare_results"])
    g.add_edge("get_data", "compute_metrics")
    g.add_conditional_edges("compute_metrics", r.after_compute, ["score_and_rank", "prepare_results"])
    g.add_edge("score_and_rank", "prepare_results")
    g.add_conditional_edges("prepare_results", r.after_prepare, ["web_context", "explain"])
    g.add_edge("web_context", "explain")
    g.add_edge("explain", "output_guard")
    g.add_conditional_edges("output_guard", r.after_guard, ["respond", "explain"])
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver
    path = get_settings().cache_dir / "conversations.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def ask(graph, thread_id: str, text: str) -> dict:
    """Run one turn. Returns the final state."""
    limit = 25
    return graph.invoke({"messages": [{"role": "user", "content": text}]},
                        config={"configurable": {"thread_id": thread_id}, "recursion_limit": limit})
