"""Grounding check and fallback answers, by answer mode (ARCHITECTURE §4.4 node 10, §8).

Answer modes (facts["answer_mode"]):
- computed: screening results from deterministic tools. Every number must come from the facts, and a number on
  a line about one airport must belong to that airport's data (no mis-attribution). Requires the assumptions
  section and the disclaimer.
- conceptual: general aviation or investment concepts. Ordinary numeric definitions are allowed. A number on a
  line about a named airport must come from the data or the question, unless the line is labelled "Illustration".
- research: external facts. Sourced figures must sit in the web section and cite a retrieved source. The guard
  checks that a source reference exists; it cannot prove a claim is supported.
- methodology: the agent's own method. Numbers come from config.

Web search claims are checked against what actually ran (facts["web"]["status"]).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from airport_agent.settings import load_config
from airport_agent.tools import public_store as ps

NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?(?:\s?[KkMm](?![a-z]))?")
CODE = re.compile(r"\b[A-Z]{3}\b")
SECTION = re.compile(r"^#{1,6}\s+(.*)$")
REQUIRED_SECTION = "assumptions & uncertainty"
DID_NOT_SEARCH = re.compile(r"searched the web|web search (found|returned|showed)|not found on the web|"
                            r"nothing (was )?found on the web|could not find (it )?on the web", re.IGNORECASE)


def _to_float(token: str) -> float:
    t = token.replace(",", "").replace("−", "-").replace("+", "").strip()
    mult = 1.0
    if t[-1] in "KkMm":
        mult = 1e3 if t[-1] in "Kk" else 1e6
        t = t[:-1].strip()
    return float(t) * mult


def numbers_in(text: str) -> list[tuple[str, float]]:
    out = []
    for m in NUM.finditer(text):
        tok = m.group(0)
        try:
            out.append((tok.strip(), _to_float(tok)))
        except ValueError:
            continue
    return out


def _allowed(*texts: str) -> set[float]:
    vals = {v for t in texts for _, v in numbers_in(t)}
    extra = set()
    for v in vals:
        extra |= {round(v, k) for k in range(3)}
        if v >= 1e3:                                   # 2,900,568 may be written 2.9M or 2,900K
            extra |= {round(v / 1e6, k) * 1e6 for k in range(3)} | {round(v / 1e3) * 1e3}
    return vals | extra


def _matches(x: float, allowed: set[float]) -> bool:
    if x in allowed or abs(x) in allowed:
        return True
    return any(abs(abs(x) - a) <= max(0.051, 0.005 * abs(a)) for a in allowed)


@lru_cache
def _airport_codes() -> frozenset[str]:
    return frozenset(ps.airports().iata.dropna())


@lru_cache
def _not_airports() -> frozenset[str]:
    return frozenset(load_config("output_guard")["acronyms_that_are_not_airports"])


def _web(facts: dict) -> dict:
    return facts.get("web") or {}


def _web_notes(facts: dict) -> list[dict]:
    return _web(facts).get("notes") or []


def _retrieved_refs(facts: dict) -> set[str]:
    """URLs and titles of the sources that were actually retrieved."""
    refs = set()
    for n in _web_notes(facts):
        for src in n.get("sources") or []:
            if src.get("url"):
                refs.add(src["url"])
            if src.get("title"):
                refs.add(src["title"])
    return refs


def _rows(facts: dict) -> list[dict]:
    """Every airport row in the facts (an airport is a dict with both 'iata' and 'airport')."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "iata" in o and "airport" in o:
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk({k: v for k, v in facts.items() if k != "web"})
    return found


def _global_facts(facts: dict) -> str:
    """Facts text without airport rows, web notes and the question: values that apply to every airport."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("web", "question") and not ("iata" in o and "airport" in o)}
        if isinstance(o, list):
            return [strip(x) for x in o if not (isinstance(x, dict) and "iata" in x and "airport" in x)]
        return o
    return json.dumps(strip(facts), default=str, ensure_ascii=False)


def _lines_with_section(draft: str):
    """Yield (line, in_web_section): lines under a heading that mentions 'web' belong to the web section."""
    in_web = False
    for line in draft.splitlines():
        h = SECTION.match(line.strip())
        if h:
            in_web = "web" in h.group(1).lower()
        yield line, in_web


COMPARATIVE_FIELDS = [("extra passengers", "forecast_extra_passengers"), ("growth", "forecast_growth"),
                      ("RPS", "RPS"), ("constraint", "C"), ("passengers", "passengers_last_12_months")]
HIGHER = re.compile(r"\b(higher|above|greater|ahead of|exceeds?|larger|more than)\b", re.IGNORECASE)
LOWER = re.compile(r"\b(lower|below|smaller|behind|trails?|less than)\b", re.IGNORECASE)
MOST = re.compile(r"\b(highest|largest|biggest|greatest|most)\b", re.IGNORECASE)
LEAST = re.compile(r"\b(lowest|smallest|least|fewest)\b", re.IGNORECASE)


def _value(text: str | None) -> float | None:
    """The leading number of a table value: '13.8%' → 13.8, '20.95M' → 20.95e6, '116K' → 116e3."""
    if not text:
        return None
    m = re.match(r"\s*([-+−]?\d[\d,]*(?:\.\d+)?)\s*([KkMm])?", str(text))
    if not m:
        return None
    v = float(m.group(1).replace(",", "").replace("−", "-"))
    return v * {"k": 1e3, "m": 1e6}.get((m.group(2) or "").lower(), 1.0)


def comparison_problems(draft: str, rows: list[dict]) -> list[str]:
    """Check comparative and superlative sentences against the airport rows (the figures the answer is built from).

    Only sentences with one measure and airport codes from the data are checked; anything else is left alone.
    'A's growth is below B's' is checked as A's growth < B's growth. 'A has the highest growth' must be the
    largest growth among the rows shown.
    """
    by_code = {r["iata"]: r for r in rows}
    problems = []
    for raw in re.split(r"(?<=[.!?])\s+|\n", draft):
        sentence = raw.strip()
        if not sentence or sentence.startswith(("|", "#")):
            continue
        codes = list(dict.fromkeys(c for c in CODE.findall(sentence) if c in by_code))
        rest, fields = sentence.lower(), []
        for phrase, field in COMPARATIVE_FIELDS:            # 'extra passengers' before 'passengers'
            if phrase.lower() in rest:
                fields.append(field)
                rest = rest.replace(phrase.lower(), " ")
        if len(set(fields)) != 1 or not codes:
            continue
        field = fields[0]
        if len(codes) == 2 and (HIGHER.search(sentence) != LOWER.search(sentence)):   # exactly one direction
            a, b = (_value(by_code[c].get(field)) for c in codes)
            if a is None or b is None:
                continue
            higher = HIGHER.search(sentence) is not None
            if (higher and not a > b) or (not higher and not a < b):
                problems.append(f'a comparison does not match the figures: "{sentence[:90]}"')
        elif len(codes) == 1 and (MOST.search(sentence) or LEAST.search(sentence)) and not (
                MOST.search(sentence) and LEAST.search(sentence)):
            values = [v for v in (_value(r.get(field)) for r in rows) if v is not None]
            mine = _value(by_code[codes[0]].get(field))
            if mine is None or not values:
                continue
            best = max(values) if MOST.search(sentence) else min(values)
            if mine != best:
                problems.append(f'a superlative does not match the figures: "{sentence[:90]}"')
    return problems


def check(draft: str, facts: dict, question: str = "") -> list[str]:
    """Return a list of problems (empty = the draft passes)."""
    mode = facts.get("answer_mode", "computed")
    web_text = " ".join(n.get("text", "") for n in _web_notes(facts))
    facts_text = json.dumps({k: v for k, v in facts.items() if k != "web"}, default=str, ensure_ascii=False)
    problems: list[str] = []

    # 1. airport codes must come from the data, the question or the retrieved sources
    known = facts_text + question + web_text + " ".join(_retrieved_refs(facts))
    codes = {c for c in CODE.findall(draft) if c in _airport_codes() and c not in _not_airports()}
    extra = sorted(c for c in codes if not re.search(rf"\b{c}\b", known))
    if extra:
        problems.append(f"airports not in the data, the question or the sources: {', '.join(extra)}")

    # 2. web search claims must match what actually ran
    status = _web(facts).get("status")
    if DID_NOT_SEARCH.search(draft) and status not in ("results", "no_reliable_result"):
        problems.append("claims a web search that did not run (or found nothing) when no search result exists")

    # 3. numbers, line by line, by section
    global_nums = _allowed(_global_facts(facts), question)
    row_nums = {r["iata"]: _allowed(json.dumps(r, default=str, ensure_ascii=False)) for r in _rows(facts)}
    web_nums = _allowed(web_text, json.dumps([src for n in _web_notes(facts) for src in n.get("sources") or []],
                                               default=str))          # dates and titles of retrieved sources
    refs = _retrieved_refs(facts)
    web_section_numbers, web_section_has_ref = [], False
    bad: set[str] = set()
    for line, in_web in _lines_with_section(draft):
        nums = numbers_in(line)
        if not nums:
            continue
        if in_web:
            if refs and any(r in draft for r in refs):
                web_section_has_ref = True
            web_section_numbers += [tok for tok, v in nums]
            bad |= {tok for tok, v in nums if not (_matches(v, web_nums) or _matches(v, global_nums))}
            continue
        line_codes = {c for c in CODE.findall(line) if c in row_nums}
        if mode == "computed":
            # a line about one airport may only carry numbers from that airport's data (or values that apply to
            # all airports); a line naming several airports may carry numbers from any of their data
            if len(line_codes) == 1:
                ok_set = row_nums[next(iter(line_codes))] | global_nums
            else:
                ok_set = global_nums | set().union(*row_nums.values()) if row_nums else global_nums
            bad |= {tok for tok, v in nums if not _matches(v, ok_set)}
        elif mode == "conceptual":
            if line_codes and not line.lstrip().lower().startswith("illustration"):
                bad |= {tok for tok, v in nums if not _matches(v, global_nums | _allowed(facts_text + question))}
        elif mode == "research":
            # a figure may come from the retrieved text (or source dates and titles), with or without a web heading
            bad |= {tok for tok, v in nums if not (_matches(v, web_nums) or _matches(v, global_nums))}
        else:                                           # methodology: numbers come from config
            bad |= {tok for tok, v in nums if not _matches(v, global_nums)}
    if bad:
        problems.append(f"numbers not in the data for their line: {', '.join(sorted(bad)[:10])}")
    if web_section_numbers and not web_section_has_ref:
        problems.append("a sourced figure has no source reference (cite a retrieved source by URL or title)")

    # 4. required sections by mode and by answer context
    if mode == "computed":
        problems += comparison_problems(draft, _rows(facts))
        # An initial answer carries the full section and the screening sentence. A focused follow-up does not repeat
        # them, but still states every caveat that is essential to this result (facts.essential_caveats).
        if not facts.get("follow_up"):
            if REQUIRED_SECTION not in draft.lower():
                problems.append('missing the section "Assumptions & uncertainty"')
            disclaimer = load_config("answer_text")["disclaimer"]
            if disclaimer.lower().rstrip(".") not in draft.lower():
                problems.append(f'missing the sentence "{disclaimer}"')
        low = draft.lower()
        for caveat in facts.get("essential_caveats") or []:
            if not any(phrase.lower() in low for phrase in caveat["accept"]):
                problems.append(f"missing a caveat needed for this answer: {caveat['text']}")
    return problems


# ---------------------------------------------------------------- fallback answers

def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(h for h, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(r.get(k) if r.get(k) is not None else "—") for _, k in cols) + " |"
                            for r in rows)


def _assumptions_block(f: dict) -> str:
    lines = [f"- Data period: {f.get('data_period', '')}"]
    lines += [f"- {a}" for a in f.get("assumptions", [])]
    lines += [f"- Scope: {s}" for s in f.get("scope_notes", [])]
    return "### Assumptions & uncertainty\n" + "\n".join(lines) + f"\n\n{f.get('disclaimer', '')}"


def web_section(facts: dict) -> str:
    """The web part of an answer, with what actually happened: results, nothing reliable, or a failed search."""
    web = _web(facts)
    if not web:
        return ""
    heading = ("### From web sources (not from the agent's data)" if facts.get("web_mode") == "lookup"
               else "### Background from web sources")
    status = web.get("status")
    if status == "failed":
        return f"{heading}\nThe web search did not run ({web.get('error') or 'service unavailable'}), so this part is missing."
    if status == "no_reliable_result" or not web.get("notes"):
        return f"{heading}\nThe web search ran but found no reliable source for this."
    out = [heading]
    for n in web["notes"]:
        out.append(n["text"])
        for src in n.get("sources") or []:
            label = src.get("title") or src.get("url")
            date = f", {src['date']}" if src.get("date") else ""
            out.append(f"- Source: [{label}]({src['url']}){date}" if src.get("url") else f"- Source: {label}{date}")
    return "\n".join(out)


def template(facts: dict, question: str = "") -> str:
    mode = facts.get("answer_mode", "computed")
    if mode == "methodology":
        return "\n\n".join([facts["summary"], facts["why_product"], facts["why_relative"], facts["why_no_spill"],
                            facts["disclaimer"]])
    if mode == "conceptual":
        return ("I can't write the full explanation right now, so I won't guess.\n\n"
                f"Your question: {facts.get('question') or question}\n\n"
                "Please ask again in a moment. The language model is the part that is unavailable; the agent's "
                "own measures (rankings, constraint scores, passenger figures) are not affected.")
    if mode == "research":
        # the fallback keeps the target the question resolved to, and carries no investment disclaimer
        target = ", ".join(a["airport"] for a in facts.get("airports") or [])
        head = f"**Airport:** {target}\n\n" if target else ""
        body = "**What the agent's data holds:** nothing on this question; the answer comes from web sources only.\n\n"
        return head + body + (web_section(facts) or "The web search did not run.")

    qt = facts.get("question_type")
    if qt in ("rank", "explain_why") and "ranking" in facts:
        by = facts.get("ranked_by", "Renovation Profitability Score (RPS)")
        scope = facts.get("scope") or facts.get("region")
        where = " across all US airports" if (scope or "").startswith("all US") else (f" in {scope}" if scope else "")
        if by.startswith("Renovation"):
            body = f"**Renovation candidates{where}** (RPS = 100 × forecast growth × C, horizon {facts['horizon_years']} years)\n\n"
            cols = [("Rank", "rank"), ("Airport", "airport"), ("Hub size", "faa_hub_size"), ("RPS", "RPS"),
                    ("RPS range (constraint only)", "RPS_range"), ("Forecast growth", "forecast_growth"), ("C", "C"),
                    ("Signals measured", "signals_measured"), ("Forecast extra passengers", "forecast_extra_passengers")]
        else:
            body = f"**Airports{where} ranked by {by}** ({facts.get('order', '')})\n\n"
            cols = [("Rank", "rank"), ("Airport", "airport"), ("Hub size", "faa_hub_size"),
                    (by[:1].upper() + by[1:], "ranked_by_value"), ("RPS", "RPS")]
            if not by.startswith("constraint"):
                cols.append(("C", "C"))
            cols.append(("Signals measured", "signals_measured"))
        body += _table(facts["ranking"], cols)
        if facts.get("flag_note"):
            body += "\n\n" + facts["flag_note"]
        if facts.get("range_meaning"):
            body += "\n\n" + facts["range_meaning"]
        if facts.get("listed_not_ranked") and by.startswith("Renovation"):
            body += "\n\n**Listed, not ranked:** " + "; ".join(
                f"{r['airport']} ({r.get('not_scored_reason') or r.get('note')})" for r in facts["listed_not_ranked"])
        caveats = {c for r in facts["ranking"] for c in r.get("caveats", [])}
    elif qt in ("compare", "explain_why") and "airports" in facts:
        rows = [{"airport": a["airport"], "C": a.get("C"), **{s["signal"]: s["value"] for s in a["constraint_signals"]}}
                for a in facts["airports"]]
        cols = [("Airport", "airport"), ("C", "C")] + [(s["signal"], s["signal"]) for s in facts["airports"][0]["constraint_signals"]]
        body = "**Constraint comparison** (C: 0 = least constrained US airport, 1 = most)\n\n" + _table(rows, cols)
        if facts.get("flag_note"):
            body += "\n\n" + facts["flag_note"]
        caveats = {c for a in facts["airports"] for c in a.get("caveats", [])}
    elif "long_haul_share_by_block_time" in facts:
        body = (f"**Long-haul share at {facts['airport']}** ({facts['source']}, {facts['period']})\n\n"
                f"- By block time ({facts['definitions']['block_time']}): {facts['long_haul_share_by_block_time']}\n"
                f"- By distance ({facts['definitions']['distance']}): {facts['long_haul_share_by_distance']}")
        if facts.get("passenger_flights"):
            body += (f"\n- Passenger flights only: {facts['passenger_long_haul_by_block_time']} by block time, "
                     f"{facts['passenger_long_haul_by_distance']} by distance")
        caveats = set(facts.get("caveats", []))
    else:
        body = "**Results**\n\n```\n" + json.dumps(facts.get("airports", facts), indent=1, default=str)[:3000] + "\n```"
        caveats = set()
    if caveats:
        body += "\n\n**Caveats:**\n" + "\n".join(f"- {c}" for c in sorted(caveats))
    # essential caveats go in the same caveats list for every context, so the fallback meets the same rules as the model
    essential = {c["text"] for c in facts.get("essential_caveats") or []}
    if essential:
        body += ("\n\n**Caveats:**\n" if not caveats else "\n") + "\n".join(f"- {c}" for c in sorted(essential - caveats))
    if _web(facts):
        body += "\n\n" + web_section(facts)
    if facts.get("follow_up"):          # a focused follow-up does not repeat the full section or the disclaimer
        return body
    return body + "\n\n" + _assumptions_block(facts)
