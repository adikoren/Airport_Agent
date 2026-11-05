"""Execution details for one turn: what ran, which calls were made, what was reused, and how the answer was checked.

Built from the graph's own stream updates (one dict per node, in execution order) and the final state of the turn.
Only facts that happened are recorded: a field is set only when it applies to this turn.
"""

from __future__ import annotations

from typing import Any

# Nodes that call Claude, with the label shown to the user.
LLM_NODES = {
    "understand_request": "Understand the question",
    "disambiguate_airports": "Resolve an ambiguous airport name",
    "web_context": "Web search (Claude with the web tool)",
    "explain": "Write the answer",
}
WEB_STATUS = {"results": "ran and returned sources",
              "no_reliable_result": "ran and found no reliable source",
              "failed": "did not run"}
STORED = ("rank", "compare", "explain_why", "metric_lookup")   # intents that can reuse results from the last answer


def build_trace(question: str, updates: list[tuple[str, dict]], final: dict[str, Any]) -> dict[str, Any]:
    """Return the per-turn snapshot: plain values only (strings, numbers, lists), ready to store and to render."""
    steps = [name for name, _ in updates]
    seen: dict[str, int] = {}
    labelled = []
    for name in steps:                      # a repeated node is a retry: explain (retry 1), output_guard (retry 1)
        seen[name] = seen.get(name, 0) + 1
        labelled.append(name if seen[name] == 1 else f"{name} (retry {seen[name] - 1})")

    counters = final.get("counters") or {}
    request = final.get("request") or {}
    intent = request.get("intent")
    trace: dict[str, Any] = {"question": question, "steps": labelled, "outcome": None}

    # ---- LLM calls (one line per call-making node that ran; retries shown by the node's own count)
    llm, explains = [], 0
    for name in steps:
        if name in LLM_NODES:
            if name == "explain":
                explains += 1
                llm.append("Write the answer" if explains == 1 else "Rewrite the answer after the number check")
            else:
                llm.append(LLM_NODES[name])
    unseen_total = int(counters.get("claude_calls", 0)) - len(llm)
    if llm:
        trace["llm_calls"] = llm + ([f"{unseen_total} more call(s) retried inside understanding"]
                                    if unseen_total > 0 else [])

    # ---- external API calls (OAG only; public data is read from local files and makes no API call)
    api = []
    for iata, lh in ((final.get("raw_data") or {}).get("long_haul") or {}).items():
        if lh.get("source") == "OAG":
            cache = "from the local cache" if lh.get("from_cache") else "live"
            api.append(f"OAG schedule for {iata}: {lh.get('calls', 0)} call(s), {cache}")
        elif lh.get("fallback_reason"):
            api.append(f"OAG not used for {iata} ({lh['fallback_reason']}); BTS domestic data used")
    if int(counters.get("api_calls", 0)) and not api:
        api.append(f"{counters['api_calls']} external API call(s)")
    if api:
        trace["api_calls"] = api
    if "get_data" in steps:
        trace["local_data"] = "Public data read from local files (BTS, FAA), no API call"

    # ---- reuse of stored results
    ran = set(steps)
    if intent in STORED and "get_data" not in ran and ran & {"prepare_results", "score_and_rank"}:
        trace["reused"] = ("Re-scored the stored measures with the changed parameters; no data fetched"
                           if "score_and_rank" in ran else
                           "Reused the results of the previous answer; no data fetched")

    # ---- web search
    status = final.get("web_status")
    if status:
        text = WEB_STATUS.get(status, status)
        if status == "failed" and final.get("web_error"):
            text += f" ({final['web_error']})"
        trace["web"] = f"Web search {text}"

    # ---- number check
    guard_updates = [w for n, w in updates if n == "output_guard"]
    rejections = []
    for i, writes in enumerate(guard_updates, 1):
        if writes.get("guard_problems"):
            action = "fixed-format fallback used" if writes.get("answer") is not None else "answer rewritten"
            rejections.append({"attempt": i, "problems": list(writes["guard_problems"]), "action": action})
    if rejections:
        trace["rejections"] = rejections

    # ---- outcome: how the final answer was produced
    fallback = any(r["action"].startswith("fixed") for r in rejections)
    not_understood = final.get("request") is None
    if final.get("errors") and not_understood:
        outcome = "error"
    elif not_understood:
        outcome = "not understood"
    elif intent == "out_of_scope":
        outcome = "scope reply"
    elif final.get("clarification"):
        outcome = "clarification"
    elif fallback:
        outcome = "fixed-format fallback"
    elif seen.get("explain", 0) > 1:
        outcome = "rewritten after the number check"
    else:
        outcome = "answer"
    trace["outcome"] = outcome
    if intent:
        trace["intent"] = intent
    if final.get("errors"):
        trace["errors"] = list(final["errors"])
    return trace
