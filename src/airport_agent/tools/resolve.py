"""Airport mentions -> airports, and region names -> airport lists (ARCHITECTURE §4.4 node 2).

Match order: exact code -> alias (config/aliases.yaml) -> exact name -> city -> fuzzy.
Only airports in the reference table are ever returned; codes are never invented.
All place-specific knowledge lives in config, never in this file.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process

from airport_agent.settings import load_config
from airport_agent.tools import public_store as ps

GENERIC = {"airport", "international", "intl", "regional", "municipal", "field", "the", "of"}
FUZZY_RESOLVE, FUZZY_CANDIDATE = 92, 80        # every plausible candidate is kept; no cap


@dataclass
class Candidate:
    airport_id: int
    iata: str
    name: str
    city: str
    state: str | None


@dataclass
class AirportRef:
    text: str
    status: str                      # resolved | ambiguous | not_found
    method: str | None = None        # exact_code | exact_name | alias | city | fuzzy | llm_context | user
    airport: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    t = t.replace("int'l", "international").replace("intl", "international")
    t = re.sub(r"\bft\.?\b", "fort", t)
    t = re.sub(r"\bmt\.?\b", "mount", t)
    t = re.sub(r"\bst\.?\b", "saint", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return " ".join(w for w in t.split() if w not in GENERIC)


@lru_cache
def _table() -> pd.DataFrame:
    """Airports that have flights in the on-time data (the ones the agent can analyse)."""
    a = ps.airports()
    ids = set(ps.hourly_cells().airport_id.unique())
    a = a[a.airport_id.isin(ids)].copy()
    a["norm_name"] = a["name"].map(normalize)
    a["norm_city"] = a["city"].fillna("").map(normalize)
    return a.reset_index(drop=True)


def _cand(row) -> Candidate:
    return Candidate(int(row.airport_id), row.iata, row["name"], row.city,
                     None if pd.isna(row.state) else row.state)


def _by_codes(codes: list[str]) -> list[Candidate]:
    t = _table()
    return [_cand(r) for _, r in t[t.iata.isin(codes)].sort_values("airport_id").iterrows()]


def resolve_one(text: str) -> AirportRef:
    t = _table()
    raw = text.strip()
    code = re.sub(r"\s+airport$", "", raw, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"[A-Za-z]{3}", code):
        hit = t[t.iata == code.upper()]
        if len(hit) == 1:
            return AirportRef(raw, "resolved", "exact_code", _cand(hit.iloc[0]))
    norm = normalize(raw)
    aliases = {normalize(k): v for k, v in load_config("aliases")["metro"].items()}
    if norm in aliases:                      # curated ambiguity first: a known ambiguous name must not silently become one airport
        cands = _by_codes(aliases[norm])
        if len(cands) == 1:
            return AirportRef(raw, "resolved", "alias", cands[0])
        if cands:
            return AirportRef(raw, "ambiguous", "alias", candidates=cands)
    hit = t[t.norm_name == norm]
    if len(hit) == 1:
        return AirportRef(raw, "resolved", "exact_name", _cand(hit.iloc[0]))
    hit = t[t.norm_city == norm]
    if len(hit) == 1:
        return AirportRef(raw, "resolved", "city", _cand(hit.iloc[0]))
    if len(hit) > 1:
        return AirportRef(raw, "ambiguous", "city", candidates=[_cand(r) for _, r in hit.iterrows()])
    best: dict[int, float] = {}
    for field_ in ("norm_name", "norm_city"):
        choices = {i: v for i, v in t[field_].items() if v}
        for _, score, i in process.extract(norm, choices, scorer=fuzz.WRatio, limit=None):
            best[i] = max(best.get(i, 0), score)
    scored = sorted(((i, s) for i, s in best.items() if s >= FUZZY_CANDIDATE), key=lambda x: (-x[1], x[0]))
    if scored and scored[0][1] >= FUZZY_RESOLVE and (len(scored) == 1 or scored[0][1] - scored[1][1] >= 5):
        return AirportRef(raw, "resolved", "fuzzy", _cand(t.loc[scored[0][0]]))
    if scored:
        return AirportRef(raw, "ambiguous", "fuzzy", candidates=[_cand(t.loc[i]) for i, _ in scored])
    return AirportRef(raw, "not_found")


def resolve_airports(texts: list[str]) -> list[AirportRef]:
    return [resolve_one(x) for x in texts]


# ---------------------------------------------------------------- regions

def region_states(region: str) -> list[str] | None:
    """Census division, Census region, state name or state code -> list of state codes. None if unknown."""
    cfg = load_config("regions")
    if is_national(region):
        return sorted(set(cfg["states"].values()))
    key = _region_key(region)
    div = {normalize(k): v for k, v in cfg["divisions"].items()}
    if key in div:
        return div[key]
    reg = {normalize(k): v for k, v in cfg["regions"].items()}
    if key in reg:
        return [s for d in reg[key] for s in cfg["divisions"][d]]
    states = {normalize(k): v for k, v in cfg["states"].items()}
    if key in states:
        return [states[key]]
    code = region.strip().upper()
    if code in set(cfg["states"].values()) and code not in {"US"}:
        return [code]
    return None


def is_national(region: str | None) -> bool:
    """True for names that mean the whole country ("US", "USA", "America", "nationwide", ...)."""
    if not region:
        return False
    names = {normalize(n) for n in load_config("regions")["national"]}
    return normalize(region) in names or normalize(region.replace(".", "")) in names


def _region_key(region: str) -> str:
    """Region name without generic words, e.g. 'the Pacific division' -> 'pacific'."""
    ignore = set(load_config("regions")["ignore_words"])
    words = [w for w in normalize(region).split() if w not in ignore]
    return " ".join(words) or normalize(region)


def expand_region(region: str) -> list[Candidate] | None:
    states = region_states(region)
    if states is None:
        return None
    t = _table()
    return [_cand(r) for _, r in t[t.state.isin(states)].sort_values("airport_id").iterrows()]


def is_metro_alias(text: str) -> bool:
    """True if the name is also a metro alias with several airports (so it needs the user's choice)."""
    aliases = {normalize(k): v for k, v in load_config("aliases")["metro"].items()}
    return len(aliases.get(normalize(text), [])) > 1
