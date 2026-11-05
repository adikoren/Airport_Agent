"""Live evaluation of scope understanding (real Claude; costs a few cents; run by hand).

    .venv/bin/python tests/evals/run_scope_eval.py

Each question has the properties a correct answer must have. The set mixes the user's reported failures
with new phrasings that appear in no prompt or other test. Prints one line per question and a summary,
and writes tests/evals/scope_eval_results.json.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.checkpoint.memory import MemorySaver

from airport_agent.graph.builder import ask, build

REPHRASE = "couldn't understand that request"
UNKNOWN_REGION = "has no official definition"

# (question, expected): intent, ranked_by (substring of the facts label), scope keyword, or "clarify"/"refuse"
CASES = [
    # the user's reported failures
    ("what is the biggest airport in US?", {"intent": "rank", "ranked_by": "passengers", "national": True}),
    ("US states", {"answered": True}),
    ("USA", {"answered": True}),
    ("what is the biggest airport in america?", {"intent": "rank", "ranked_by": "passengers", "national": True}),
    ("most constraint airportss", {"intent": "rank", "ranked_by": "constraint"}),
    ("Rank the airports in the Pacific division", {"intent": "rank", "scope": "Pacific"}),
    # new phrasings (held out: not in any prompt or other test)
    ("which airports have the fullest planes nationwide?", {"intent": "rank", "ranked_by": "load factor", "national": True}),
    ("smallest commercial airport in Texas", {"intent": "rank", "ranked_by": "passengers", "order": "lowest"}),
    ("top 5 fastest growing airports in the south", {"intent": "rank", "ranked_by": "growth", "shown": 5}),
    ("where do planes wait longest for a gate?", {"intent": "rank", "ranked_by": "taxi-in"}),
    ("best renovation candidates in Florida", {"intent": "rank", "ranked_by": "Renovation"}),
    ("how busy is Denver airport", {"answered": True}),
    ("compare Atlanta and Charlotte congestion", {"intent": "compare"}),
    ("which midwest airport has the worst delays", {"intent": "rank", "ranked_by": "delay"}),
    ("what's the weather in Boston", {"intent": "out_of_scope"}),
    ("rank London airports", {"intent": "out_of_scope"}),
    ("rank west coast airports", {"clarify": True}),
]


# A second set written after the prompts were generalised: wording the prompts have never seen.
UNSEEN = [
    ("which Ohio airports could use more gate space?", {"intent": "rank", "ranked_by": "gate"}),
    ("show me the five airports where demand is growing fastest", {"intent": "rank", "ranked_by": "growth", "shown": 5}),
    ("how full are planes at Phoenix", {"intent": "metric_lookup"}),
    ("is Tulsa more crowded than Oklahoma City?", {"intent": "compare"}),
    ("what share of Honolulu flights go far?", {"intent": "metric_lookup"}),
    ("name the quietest airports in the whole country", {"intent": "rank", "ranked_by": "constraint", "order": "lowest", "national": True}),
    ("why does Newark rank so poorly?", {"intent": "explain_why"}),
    ("what's the biggest airport in Canada?", {"intent": "out_of_scope"}),
    ("tell me the history of JFK terminal 1", {"intent": "research"}),
    ("which mountain state airport handles the most passengers?", {"intent": "rank", "ranked_by": "passengers"}),
]


def check(state: dict, exp: dict) -> list[str]:
    text = state["messages"][-1]["content"]
    req = state.get("request") or {}
    f = state.get("facts") or {}
    problems = []
    if REPHRASE in text:
        problems.append("answered 'please rephrase'")
    if exp.get("clarify"):
        if UNKNOWN_REGION not in text and "Which" not in text:
            problems.append("expected a clarifying question")
        return problems
    if exp.get("answered") and (UNKNOWN_REGION in text):
        problems.append("treated a valid place as unknown")
    if "intent" in exp and req.get("intent") != exp["intent"]:
        problems.append(f"intent {req.get('intent')} (expected {exp['intent']})")
    if "ranked_by" in exp and exp["ranked_by"].lower() not in (f.get("ranked_by") or "").lower():
        problems.append(f"ranked by {f.get('ranked_by')!r} (expected {exp['ranked_by']!r})")
    if exp.get("national") and "all US airports" not in (f.get("scope") or ""):
        problems.append(f"scope {f.get('scope')!r} (expected the whole country)")
    if "scope" in exp and exp["scope"].lower() not in (f.get("scope") or "").lower():
        problems.append(f"scope {f.get('scope')!r}")
    if "order" in exp and exp["order"] not in (f.get("order") or ""):
        problems.append(f"order {f.get('order')!r}")
    if "shown" in exp and f.get("airports_shown") != exp["shown"]:
        problems.append(f"shown {f.get('airports_shown')} (expected {exp['shown']})")
    if UNKNOWN_REGION in text and not exp.get("clarify"):
        problems.append("asked about an unknown region")
    return problems


def main() -> None:
    graph = build(MemorySaver())
    results, ok = [], 0
    cases = CASES + UNSEEN if "--unseen" in sys.argv else CASES
    for q, exp in cases:
        t = time.time()
        state = ask(graph, str(uuid.uuid4()), q)
        problems = check(state, exp)
        ok += not problems
        f = state.get("facts") or {}
        top = [r.get("iata") for r in (f.get("ranking") or [])[:3]]
        print(f"{'PASS' if not problems else 'FAIL'} {time.time() - t:4.0f}s  {q[:52]:52} "
              f"{(state.get('request') or {}).get('intent')!s:14} {top} {'; '.join(problems)}")
        results.append({"question": q, "problems": problems, "intent": (state.get("request") or {}).get("intent"),
                        "ranked_by": f.get("ranked_by"), "scope": f.get("scope"), "top": top,
                        "answer": state["messages"][-1]["content"]})
    print(f"\n{ok}/{len(cases)} passed")
    (ROOT / "tests/evals/scope_eval_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
