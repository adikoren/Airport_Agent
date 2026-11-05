"""A scripted stand-in for Claude, for offline tests (AIRPORT_AGENT_CLAUDE=fake).

It reads requests with simple patterns (never with airport names from the tests), picks candidates only when the
text names one of them, and "writes" the answer with the deterministic fallback, so every number still comes from
code. It is not a model: the live evaluation (tests/evals/run_scope_eval.py) is the check of understanding.
"""

from __future__ import annotations

import re

from airport_agent.llm.schemas import AirportChoice, Disambiguation, ParsedRequest
from airport_agent.tools import output_guard

STOP = {"Which", "What", "Compare", "How", "Why", "Rank", "And", "Use", "Is", "The", "Tell", "Show", "Who", "When"}


def _places(text: str) -> list[str]:
    """Capitalised phrases (a city or airport name) and 2–3 letter codes."""
    found = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,3})\b", text)
    return [p for p in found if p not in STOP and p.lower() not in {"airport", "airports"}]


class FakeClaude:
    def understand(self, history: list[dict], context_summary: str) -> ParsedRequest:
        text = history[-1]["content"]
        low = text.lower()
        base = dict(follow_up=False, top_n=None)
        if "how do you score" in low or "methodology" in low:
            return ParsedRequest(intent="methodology", **base)
        if re.search(r"\birr\b", low) and re.search(r"\b(forecast|project|projection|cash|return)", low):
            return ParsedRequest(intent="out_of_scope", **base)
        if "first one first" in low:
            return ParsedRequest(intent="explain_why", follow_up=True, top_n=None)
        if m := re.search(r"(\d+)[- ]year", low):
            return ParsedRequest(intent="rank", follow_up=True, top_n=None, horizon_years=int(m.group(1)))
        if m := re.search(r"\bin (?:the )?([A-Z][\w ]+?) (?:are|is|airports)\b", text):
            return ParsedRequest(intent="rank", region=m.group(1).strip(), **base)
        if any(w in low for w in ("who owns", "owner", "history", "budget", "planned", "how many gates", "built")):
            return ParsedRequest(intent="research", airports=_places(text)[:2], **base)
        if low.startswith(("what is", "what are", "what does", "explain", "define", "how does")) and not _places(text):
            return ParsedRequest(intent="conceptual", **base)
        if "long haul" in low:
            return ParsedRequest(intent="metric_lookup", airports=_places(text)[:1], metrics=["long_haul_share"], **base)
        if "why" in low or "unmet" in low:
            return ParsedRequest(intent="explain_why", airports=_places(text)[:1], **base)
        if low.startswith("compare") or "congest" in low:
            return ParsedRequest(intent="compare", airports=_places(text), **base)
        return ParsedRequest(intent="out_of_scope", **base)

    def disambiguate(self, history: list[dict], options: dict[str, list[dict]]) -> Disambiguation:
        """Choose only when another candidate is named separately in the message; otherwise ask."""
        text = history[-1]["content"].lower()
        choices = []
        for mention, cands in options.items():
            named_elsewhere = any(c["city"] and c["city"].lower() in text and c["city"].lower() not in mention.lower()
                                  for c in cands)
            rest = [c for c in cands if not (c["city"] and c["city"].lower() in text)]
            if named_elsewhere and rest and len({c["state"] for c in cands}) == 1:
                # test stub only: the largest candidate that was not named separately (the live model reasons instead)
                pick = max(rest, key=lambda c: _people(c.get("passengers_last_12_months")))
                choices.append(AirportChoice(mention=mention, choice=pick["code"], confidence="high"))
            else:
                choices.append(AirportChoice(mention=mention, choice=None, confidence="low"))
        return Disambiguation(choices=choices)

    def web_context(self, question: str, airports: list[str]) -> list[dict]:
        return [{"text": "- (offline test) background notes would appear here.",
                 "sources": [{"url": "https://example.org/offline-test", "title": "Offline test source", "date": None}]}]

    def web_lookup(self, question: str, airports: list[str], gaps: list[str]) -> list[dict]:
        return [{"text": "- (offline test) a web answer would appear here.",
                 "sources": [{"url": "https://example.org/offline-test", "title": "Offline test source", "date": "2026"}]}]

    def explain(self, history: list[dict], facts: dict, problems: list[str] | None = None) -> str:
        return output_guard.template(facts, history[-1]["content"] if history else "")


def _people(text: str | None) -> float:
    if not text:
        return 0.0
    t = text.replace(",", "")
    mult = 1e6 if t.endswith("M") else 1e3 if t.endswith("K") else 1
    return float(t.rstrip("MK")) * mult
