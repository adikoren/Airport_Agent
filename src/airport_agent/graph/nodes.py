"""The 11 nodes (ARCHITECTURE §4.4). Code nodes are thin wrappers around tools; four nodes call Claude.

A node returns only the state keys it owns (see state.NODE_WRITES).
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime

import pandas as pd

from airport_agent.graph.state import AgentState, empty_counters
from airport_agent.llm.client import Claude, ClaudeUnavailable
from airport_agent.settings import get_settings, load_config
from airport_agent.tools import facts as F
from airport_agent.tools import metrics as m
from airport_agent.tools import output_guard as guard
from airport_agent.tools import public_store as ps
from airport_agent.tools import resolve as rs
from airport_agent.tools import scoring
from airport_agent.tools.scoring import (
    AirportScore,
    compute_signals,
    rank,
    score_airport,
)

SCORED_INTENTS = {"rank"}           # only rankings compute RPS; "why" and comparisons use the drivers
_claude: Claude | None = None


def claude() -> Claude:
    """One Claude client per process; tests replace it with set_claude()."""
    global _claude
    if _claude is None:
        if os.environ.get("AIRPORT_AGENT_CLAUDE") == "fake":     # offline end-to-end tests only
            from airport_agent.llm.fake import FakeClaude
            _claude = FakeClaude()
        elif get_settings().llm_provider == "anthropic":
            _claude = Claude()
        else:                                                     # default gateway
            from airport_agent.llm.openrouter import OpenRouterLLM
            _claude = OpenRouterLLM()
    return _claude


def set_claude(c) -> None:
    global _claude
    _claude = c


def _history(state: AgentState, n: int = 10) -> list[dict]:
    return [{"role": x["role"], "content": x["content"]} for x in state.get("messages", [])[-n:]]


def _context_summary(ctx: dict) -> str:
    if not ctx:
        return "No earlier results in this conversation."
    req = ctx.get("request", {})
    return json.dumps({"previous_request_type": req.get("intent"), "previous_region": req.get("region"),
                       "previous_airports": ctx.get("airport_codes"), "previous_parameters": ctx.get("params")})


# 1 ------------------------------------------------------------------- understand_request (Claude)

def understand_request(state: AgentState) -> dict:
    counters = empty_counters()
    reset = {"airport_refs": [], "clarification": None, "facts": {}, "scores": [], "web_context": [],
             "web_status": None, "web_error": None,
             "draft_answer": None, "answer": None, "guard_problems": [], "raw_data": {}, "data_quality": {},
             "metrics": {}, "errors": []}
    last_error = None
    for _ in range(2):                                    # first try + 1 retry
        counters["claude_calls"] += 1
        try:
            parsed = claude().understand(_history(state), _context_summary(state.get("context", {})))
            message = (state.get("messages") or [{}])[-1].get("content", "")
            req = normalize_request(parsed.model_dump(), message)
            req, clarification = resolve_ranked_positions(req, state.get("context") or {})
            return {**reset, "request": req, "counters": counters, "clarification": clarification}
        except ClaudeUnavailable as e:
            last_error = str(e)
    return {**reset, "request": None, "counters": counters, "errors": [f"understand_request: {last_error}"]}


REFERS_TO_AIRPORTS = {"research", "explain_why", "compare", "metric_lookup"}


def resolve_ranked_positions(req: dict, ctx: dict) -> tuple[dict, dict | None]:
    """Resolve 'the top-ranked airport' (or 'the second one') from the latest stored ranking, in code, before any
    lookup or web search. Returns the request, and a clarification when the reference cannot be resolved."""
    positions = req.get("ranked_positions") or []
    if not positions or req.get("intent") not in REFERS_TO_AIRPORTS:
        return req, None
    ranked = sorted((s for s in ctx.get("scores") or [] if s.get("rank")), key=lambda s: s["rank"])
    if not ranked:
        return req, {"text": "I don't have a ranking in this conversation yet. Ask me to rank airports first, "
                              "or name the airport."}
    chosen = [ranked[p - 1]["iata"] for p in positions if 1 <= p <= len(ranked)]
    if not chosen:
        return req, {"text": f"The last ranking has {len(ranked)} ranked airports, so position {positions[0]} is not "
                              "in it. Name the airport."}
    # A mention like "the top-ranked airport" names no airport. Left in, it would go to disambiguation, where a
    # model could pick some unrelated airport. Keep only mentions that resolve to a real airport.
    from airport_agent.tools import resolve as R
    named = [m for m in req.get("airports") or [] if R.resolve_one(m).status != "not_found"]
    req = dict(req)
    req["airports"] = list(dict.fromkeys(chosen + named))
    req["resolved_from_ranking"] = chosen
    return req, None


RANKABLE = {"rps", "constraint", "passengers", "growth_share", "load_factor", "peak_saturation",
            "taxi_out_excess", "taxi_in_excess", "international_share", "delay_rate", "cancellation_rate"}
NATIONAL = "United States"


TOP_N = re.compile(r"\btop\s+(\d{1,3})\b", re.IGNORECASE)


def normalize_request(req: dict, message: str = "") -> dict:
    """Code rules applied to Claude's parse (scope rules, ARCHITECTURE §4.2):
    - a metric asked for a whole scope (no named airports) is a ranking by that metric;
    - a ranking with no place named covers all US airports;
    - "top N" written in the message sets top_n if Claude left it empty."""
    req = dict(req)
    if req.get("top_n") is None and (m_ := TOP_N.search(message)):
        req["top_n"] = int(m_.group(1))
    if req["intent"] == "metric_lookup" and not req.get("airports") and req.get("metrics") \
            and req["metrics"][0] in RANKABLE:
        req.update(intent="rank", rank_by=req["metrics"][0], metrics=[])
    if req["intent"] == "rank" and not req.get("airports") and not req.get("region") and not req.get("follow_up"):
        req["region"] = NATIONAL
    return req


# 2 ------------------------------------------------------------------- resolve_airports

def resolve_airports(state: AgentState) -> dict:
    req = state["request"]
    refs: list[dict] = []
    region = req.get("region")
    if region and rs.is_metro_alias(region):
        # e.g. a name that is both a state and a metro area: ask instead of guessing
        metro = rs.resolve_one(region)
        listed = ", ".join(f'{c.name} ({c.iata})' for c in metro.candidates)
        n_state = len(rs.expand_region(region) or [])
        text = (f'Before I answer, one check: by "{region}", do you mean the metro area ({listed}) '
                f'or the whole state ({n_state} airports with flight data)?')
        return {"airport_refs": [], "clarification": {"text": text}}
    if region:
        members = rs.expand_region(region)
        if members is None:
            text = load_config("answer_text")["unknown_region_reply"].format(region=region)
            return {"airport_refs": [], "clarification": {"text": text}}
        refs += [rs.AirportRef(region, "resolved", "region", c).to_dict() for c in members]
    mentions = [x for x in (req.get("airports") or [])
                if not (region and rs.normalize(region) in rs.normalize(x))]   # a mention that just repeats the region
    if not mentions and not region and req.get("follow_up"):
        mentions = state.get("context", {}).get("airport_codes", [])
    refs += [r.to_dict() for r in rs.resolve_airports(mentions)]
    return {"airport_refs": refs, "clarification": None}


# 3 ------------------------------------------------------------------- disambiguate_airports (Claude)

def disambiguate_airports(state: AgentState) -> dict:
    refs = [dict(r) for r in state["airport_refs"]]
    counters = dict(state.get("counters") or empty_counters())
    open_refs = [r for r in refs if r["status"] == "ambiguous"]
    choices = {}
    if open_refs:
        ref = ps.airports().set_index("iata")
        period, t100 = ps.default_period(), ps.t100_monthly()

        def info(c):
            pax = m.passengers(t100, c["airport_id"], period).value
            hub = ref.loc[c["iata"], "hub_size"] if c["iata"] in ref.index else None
            return {"code": c["iata"], "name": c["name"], "city": c["city"], "state": c["state"],
                    "faa_hub_size": None if hub is None or hub != hub else hub,
                    "passengers_last_12_months": None if pax is None else F.people(pax)}
        options = {r["text"]: [info(c) for c in r["candidates"]] for r in open_refs}
        counters["claude_calls"] += 1
        try:
            result = claude().disambiguate(_history(state), options)
            choices = {c.mention: c for c in result.choices}
        except ClaudeUnavailable:
            choices = {}
    ask = []
    for r in refs:
        if r["status"] == "not_found":
            ask.append(f'- "{r["text"]}": I couldn\'t find this airport. Could you give its name or code?')
        elif r["status"] == "ambiguous":
            c = choices.get(r["text"])
            codes = {x["iata"]: x for x in r["candidates"]}
            if c and c.choice in codes and c.confidence == "high":       # checked in code, not trusted
                r.update(status="resolved", method="llm_context", airport=codes[c.choice])
            else:
                listed = ", ".join(f'{x["name"]} ({x["iata"]})' for x in r["candidates"])
                ask.append(f'- "{r["text"]}" could be: {listed}. Which one do you mean?')
    clarification = {"text": "Before I answer, one check:\n" + "\n".join(ask)} if ask else None
    return {"airport_refs": refs, "clarification": clarification, "counters": counters}


# 4 ------------------------------------------------------------------- get_data

def _airports(state: AgentState) -> list[dict]:
    seen, out = set(), []
    for r in state["airport_refs"]:
        a = r.get("airport")
        if r["status"] == "resolved" and a and a["airport_id"] not in seen:
            seen.add(a["airport_id"])
            out.append(a)
    return out


def get_data(state: AgentState) -> dict:
    """Plan + fetch + hard checks. Public data is local; OAG is called only for long haul."""
    req = state["request"]
    counters = dict(state.get("counters") or empty_counters())
    raw, quality, errors = {}, {}, []
    period = ps.default_period()
    t100 = ps.t100_monthly()
    for a in _airports(state):
        months = m.t100_window(t100, a["airport_id"], period).month.nunique()
        reason = None if months == 12 else f"only {months} of 12 months in BTS T-100"
        quality[a["iata"]] = {"status": "ok" if reason is None else "missing", "reason": reason}
    if req["intent"] == "metric_lookup" and "long_haul_share" in (req.get("metrics") or []):
        raw["long_haul"] = {}
        date = load_config("scoring")["long_haul"]["oag_reference_date"]
        for a in _airports(state)[:1]:                         # one airport per question (free-tier budget)
            entry = {"date": date}
            try:
                from airport_agent.tools.oag_client import OAGClient
                res = OAGClient().departures(a["iata"], date)
                counters["api_calls"] += res.calls
                entry.update(source="OAG", flights=res.flights, from_cache=res.from_cache, calls=res.calls)
            except Exception as e:                            # budget, auth, network: fall back to BTS
                entry.update(source="BTS", fallback_reason=str(e))
                errors.append(f"OAG long haul for {a['iata']}: {e}")
            raw["long_haul"][a["iata"]] = entry
    return {"raw_data": raw, "data_quality": quality, "counters": counters, "errors": errors}


# 5 ------------------------------------------------------------------- compute_metrics

def compute_metrics(state: AgentState) -> dict:
    period = ps.default_period()
    out = {}
    for a in _airports(state):
        sig = compute_signals(a["airport_id"], period)
        row = {k: v.to_dict() for k, v in sig.items()}
        row.update({k: v.to_dict() for k, v in m.delay_and_cancellation(ps.ontime_monthly(), a["airport_id"], period).items()})
        row["passengers"] = m.passengers(ps.t100_monthly(), a["airport_id"], period).to_dict()
        row["international_share"] = m.international_share(ps.t100_monthly(), a["airport_id"], period).to_dict()
        lh = state.get("raw_data", {}).get("long_haul", {}).get(a["iata"])
        if lh is not None:
            if lh["source"] == "OAG":
                row["long_haul_share"] = m.long_haul_from_flights(lh["flights"], f"OAG schedule, {lh['date']}").to_dict()
            else:
                row["long_haul_share"] = m.long_haul_bts(ps.ontime_monthly(), a["airport_id"], period).to_dict()
                row["long_haul_share"]["caveats"].append(f"OAG not used: {lh.get('fallback_reason')}")
        out[a["iata"]] = row
    return {"metrics": out}


# 6 ------------------------------------------------------------------- score_and_rank

def _params(req: dict, ctx: dict | None = None) -> dict:
    """Scoring parameters: stated ones win; a follow-up inherits the rest; a new question uses defaults."""
    cfg = load_config("scoring")
    weights = {k: v for k, v in (req.get("weights") or {}).items() if v is not None} or None
    prev = (ctx or {}).get("params", {}) if req.get("follow_up") else {}
    return {"horizon": req.get("horizon_years") or prev.get("horizon") or cfg["horizon"]["default_years"],
            "weights": weights or prev.get("weights"),
            "rank_by": req.get("rank_by") or prev.get("rank_by") or "rps",
            "order": req.get("order") or prev.get("order") or "highest",
            "top_n": req.get("top_n") or prev.get("top_n")}


SHOW_ALL_UP_TO, SHOW_TOP = 20, 10       # presentation: list every airport in small scopes, the top 10 in large ones


def shown_count(n_ranked: int, top_n: int | None) -> int:
    return top_n or (n_ranked if n_ranked <= SHOW_ALL_UP_TO else SHOW_TOP)


def _mv(d: dict) -> m.MetricValue:
    return m.MetricValue(**d)


def score_and_rank(state: AgentState) -> dict:
    req = state["request"]
    ctx = state.get("context", {})
    metrics_ = state.get("metrics") or ctx.get("metrics", {})
    params = _params(req, ctx)
    ids = {a["iata"]: a["airport_id"] for a in _airports(state)} if state.get("airport_refs") else ctx.get("airport_ids", {})
    sigs = {iata: {k: _mv(metrics_[iata][k]) for k in ("load_factor", "peak_saturation", "taxi_out_excess",
                                                      "taxi_in_excess")} for iata in ids}
    try:      # score every airport without the (slower) range first
        scores = [score_airport(ap_id, horizon=params["horizon"], user_weights=params["weights"],
                                signals=sigs[iata], with_range=False) for iata, ap_id in ids.items()]
    except ValueError as e:
        return {"scores": [], "errors": [f"score_and_rank: {e}"]}
    extra = {iata: {k: (metrics_[iata].get(k) or {}).get("value") for k in ("delay_rate", "cancellation_rate")}
             for iata in ids}
    ranked = rank(scores, params["rank_by"], params["order"], extra)
    n_show = shown_count(sum(1 for s in ranked if s.rank), params["top_n"])
    for i, s in enumerate(ranked):        # the measured range only for the airports that will be shown
        if s.rank and s.rank <= n_show and s.scored:
            full = score_airport(s.airport_id, horizon=params["horizon"], user_weights=params["weights"],
                                 signals=sigs[s.iata], with_range=True)
            full.rank, full.rank_value = s.rank, s.rank_value
            ranked[i] = full
    return {"scores": [s.to_dict() for s in ranked]}


# 7 ------------------------------------------------------------------- prepare_results

def _scores(state: AgentState) -> list[AirportScore]:
    rows = state.get("scores") or state.get("context", {}).get("scores", [])
    return [AirportScore(**{**r, "growth_years": tuple(r["growth_years"]) if r.get("growth_years") else None,
                            "C_range": tuple(r["C_range"]) if r.get("C_range") else None}) for r in rows]


SIGNAL_KEYS = ("load_factor", "peak_saturation", "taxi_out_excess", "taxi_in_excess")


def _question(state: AgentState) -> str:
    return (state.get("messages") or [{}])[-1].get("content", "")


def _driver_rows(state: AgentState, metrics_: dict, horizon: int) -> list[dict]:
    """Constraint, growth and size for each airport in the question. No RPS."""
    period = ps.default_period()
    ref = ps.airports().set_index("airport_id")
    rows = []
    for a in _airports(state):
        r = ref.loc[a["airport_id"]]
        sig = {k: _mv(metrics_[a["iata"]][k]) for k in SIGNAL_KEYS}
        pax = _mv(metrics_[a["iata"]]["passengers"]).value
        d = scoring.drivers(sig, None if pd.isna(r.faa_id) else r.faa_id, pax, period, horizon)
        hub = None if pd.isna(r.hub_size) else r.hub_size
        rows.append(F.driver_row(a["name"], a["iata"], a.get("state"), hub, d))
    return rows


def prepare_results(state: AgentState) -> dict:
    """Build the facts, then mark whether this is a focused follow-up and which caveats the answer must state."""
    out = _prepare_facts(state)
    facts = out.get("facts")
    if facts is not None and facts.get("answer_mode", "computed") == "computed":
        facts = {**facts, "follow_up": bool((state.get("request") or {}).get("follow_up")),
                 "essential_caveats": F.essential_caveats(facts)}
        out = {**out, "facts": facts}
    return out


def _prepare_facts(state: AgentState) -> dict:
    """Build the facts for this answer mode. Only rankings score; comparisons and 'why' use drivers."""
    req = state["request"]
    ctx = state.get("context", {})
    period = ps.default_period()
    label = period.label
    notes = [f"not covered: {p}" for p in req.get("out_of_scope_parts") or []]
    intent = req["intent"]
    question = _question(state)
    if intent == "conceptual":
        return {"facts": F.conceptual_facts(question, notes)}
    if intent == "methodology":
        return {"facts": F.methodology_facts()}
    if intent == "research":
        facts = F.research_facts(question, _airports(state), notes)
        facts["web_mode"] = "lookup"
        return {"facts": facts}
    params = _params(req, ctx)
    region = req.get("region") or ctx.get("request", {}).get("region")
    if intent == "rank":
        facts = F.ranking_facts(_scores(state), region, label, params["horizon"], notes, params)
        return {"facts": facts}
    metrics_ = state.get("metrics") or ctx.get("metrics", {})
    if intent == "explain_why" and (state.get("scores") or (req.get("follow_up") and ctx.get("scores"))):
        facts = F.ranking_facts(_scores(state), region, label, params["horizon"], notes, params)
        facts["question_type"] = "explain_why"            # explain results already computed; nothing new is scored
        return {"facts": facts}                           # answered from the stored ranking inputs: no web search
    if intent in ("compare", "explain_why"):
        rows = _driver_rows(state, metrics_, params["horizon"])
        facts = F.drivers_facts(rows, intent, label, params["horizon"], notes)
        if intent == "explain_why":
            facts["web_mode"] = "context"
        return {"facts": facts}
    # metric_lookup
    airports = _airports(state)
    if "long_haul_share" in (req.get("metrics") or []) and airports:
        a = airports[0]
        mv = _mv(metrics_[a["iata"]]["long_haul_share"])
        note = ("OAG schedule for one day, operating flights only (codeshare copies removed), domestic and "
                "international" if mv.source == "OAG" else "BTS on-time data, domestic flights only")
        return {"facts": F.long_haul_facts(a["iata"], a["name"], mv, note, label, notes)}
    rows, gaps = [], []
    for a in airports:
        r = {"airport": f"{a['name']} ({a['iata']})"}
        for k in req.get("metrics") or []:
            v = metrics_[a["iata"]].get(k)
            if v and v.get("value") is not None:
                r[k] = F.pct(v["value"]) if v["unit"] == "share" else F.people(v["value"]) if v["unit"] == "passengers" \
                    else F.num(v["value"])
            else:          # the tools couldn't provide it: the web research may
                reason = (v or {}).get("missing_reason") or "not in the agent's data"
                r[f"{k}_missing"] = reason
                gaps.append(f"{a['iata']} {k.replace('_', ' ')} ({reason})")
        rows.append(r)
    facts = F.metric_facts(rows, label, notes)
    if gaps:
        facts["data_gaps"] = gaps
        facts["web_mode"] = "lookup"
    return {"facts": facts}


# 8 ------------------------------------------------------------------- web research (Claude web search)

def web_context(state: AgentState) -> dict:
    """Web research for the mode the facts ask for: 'lookup' (external facts) or 'context' (background).

    The outcome is recorded as web_status: results, no_reliable_result, or failed (the search did not run).
    """
    counters = dict(state.get("counters") or empty_counters())
    mode = (state.get("facts") or {}).get("web_mode")
    if not mode:
        return {}
    question = _question(state)
    codes = [r["iata"] for r in state["facts"].get("ranking", [])][:3] or [a["iata"] for a in _airports(state)][:3]
    counters["claude_calls"] += 1
    try:
        if mode == "lookup":
            notes = claude().web_lookup(question, codes, (state["facts"].get("data_gaps") or []))
        else:
            notes = claude().web_context(question, codes)
        status, error = ("results" if notes else "no_reliable_result"), None
    except ClaudeUnavailable as e:
        notes, status, error = [], "failed", str(e)
    return {"web_context": [dict(n, kind=mode) for n in notes], "web_status": status, "web_error": error,
            "counters": counters}


def _attach_web(state: AgentState) -> dict:
    """The facts plus this turn's web outcome, for explain and the number check."""
    facts = dict(state["facts"])
    if state.get("web_status") is not None:
        facts["web"] = {"status": state.get("web_status"), "error": state.get("web_error"),
                        "notes": state.get("web_context") or []}
    return facts


# 9 ------------------------------------------------------------------- explain (Claude)

def explain(state: AgentState) -> dict:
    counters = dict(state.get("counters") or empty_counters())
    facts = _attach_web(state)
    problems = state.get("guard_problems") or None
    counters["claude_calls"] += 1
    if problems:
        counters["explain_retries"] += 1
    try:
        draft = claude().explain(_history(state), facts, problems)
    except ClaudeUnavailable as e:
        return {"draft_answer": None, "counters": counters, "errors": [f"explain: {e}"]}
    return {"draft_answer": draft, "counters": counters}


# 10 ------------------------------------------------------------------ output_guard

def output_guard(state: AgentState) -> dict:
    counters = dict(state.get("counters") or empty_counters())
    facts = _attach_web(state)
    draft = state.get("draft_answer")
    if draft:
        problems = guard.check(draft, facts, _question(state))
        if not problems:
            return {"answer": draft, "guard_problems": [], "counters": counters}
        if counters["explain_retries"] < load_config("scoring")["limits"]["max_explain_retries"]:
            return {"answer": None, "guard_problems": problems, "counters": counters}
    else:
        problems = ["no draft from Claude"]
    note = "\n\n_(Shown as a fixed-format answer: the written version didn't pass the checks.)_"
    return {"answer": guard.template(facts, _question(state)) + note, "guard_problems": problems, "counters": counters}


# 11 ------------------------------------------------------------------ respond

def respond(state: AgentState) -> dict:
    text_cfg = load_config("answer_text")
    req = state.get("request")
    if req is None:
        errors = state.get("errors", [])
        text = text_cfg["rephrase_reply"]
        if missing := next((k for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY") if any(k in e for e in errors)), None):
            text = f"The language model isn't configured: add {missing} to .env and restart."
        elif problem := next((e.split(": ", 1)[1] for e in errors if "API unavailable" in e), None):
            # the question was fine; the service wasn't: say so instead of asking the user to rephrase
            text = f"I can't answer right now ({problem}). Your question is fine; please try again later."
    elif req["intent"] == "out_of_scope":
        text = text_cfg["scope_reply"]
    elif state.get("clarification"):
        text = state["clarification"]["text"]
    else:
        text = state.get("answer") or text_cfg["rephrase_reply"]
    ctx = dict(state.get("context") or {})
    if state.get("scores") or state.get("metrics"):
        if req and ctx.get("scores") and req["intent"] != "rank":
            # A turn that discusses or compares airports does not change the ranking population, its metrics or
            # its scores. Keep the earlier context as it is, so a later ranking follow-up sees the full population.
            pass
        else:
            airports = _airports(state) if state.get("airport_refs") else []
            region = req.get("region") if req else None
            if req and req.get("follow_up") and not region:
                region = ctx.get("request", {}).get("region")        # a follow-up inherits the earlier place
            ctx = {"request": {**req, "region": region} if req else None,
                   "airport_codes": [a["iata"] for a in airports] or ctx.get("airport_codes", []),
                   "airport_ids": {a["iata"]: a["airport_id"] for a in airports} or ctx.get("airport_ids", {}),
                   "metrics": state.get("metrics") or ctx.get("metrics", {}),
                   "scores": state.get("scores") or ctx.get("scores", []),
                   "params": _params(req, ctx) if req else ctx.get("params", {}),
                   "period": ps.default_period().label}
    _audit(state, text)
    return {"messages": [{"role": "assistant", "content": text}], "context": ctx}


def _audit(state: AgentState, text: str) -> None:
    """One JSON line per turn in data/cache/audit.jsonl: what was asked, what ran, what was answered."""
    req = state.get("request") or {}
    entry = {"time": datetime.now(UTC).isoformat(timespec="seconds"),
             "question": (state.get("messages") or [{}])[-1].get("content"),
             "intent": req.get("intent"), "region": req.get("region"), "airports": req.get("airports"),
             "resolved": [(r.get("airport") or {}).get("iata") for r in state.get("airport_refs") or []][:30],
             "counters": state.get("counters"), "guard_problems": state.get("guard_problems"),
             "errors": state.get("errors"), "clarification": bool(state.get("clarification")),
             "answer_chars": len(text)}
    try:
        path = get_settings().cache_dir / "audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass  # the audit log must never break an answer
