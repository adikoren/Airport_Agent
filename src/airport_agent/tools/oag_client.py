"""OAG Flight Info client — long-haul share only (decision D2, docs/oag_notes.md).

- Auth: `Subscription-Key` header. The key is sent only to api.oag.com and never logged.
- Paging: follow `paging.next` until it is absent. A `next` URL on any other host is refused.
- Free tier: 10 records per call, so one airport-day costs 49–177 calls. A per-question budget
  stops the fetch before it starts if the day doesn't fit (the caller then falls back to BTS).
- Operating flights only: records with `codeshare.operatingFlight` are marketing copies and are dropped.
- Responses are cached on disk (data/cache/oag), so a repeated question makes no calls.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx

from airport_agent.settings import get_settings, load_config

API_HOST = "api.oag.com"
BASE_URL = f"https://{API_HOST}/flight-instances/"


class OAGError(RuntimeError):
    """An OAG failure, with a message that never contains the key."""


class BudgetExceeded(OAGError):
    pass


@dataclass
class FetchResult:
    flights: list[dict]
    calls: int
    from_cache: bool
    total_records: int
    notes: list[str] = field(default_factory=list)


def _safe(msg: str, key: str) -> str:
    return msg.replace(key, "***") if key else msg


def _cache_path(airport: str, date: str):
    d = get_settings().cache_dir / "oag"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{airport}_{date}.json"


def operating_flight(record: dict) -> dict | None:
    """Flatten one OAG record into the fields long haul needs; None for a codeshare copy."""
    if (record.get("codeshare") or {}).get("operatingFlight"):
        return None
    minutes = record.get("elapsedTime")
    dep, arr = record.get("departure") or {}, record.get("arrival") or {}
    return {
        "carrier": (record.get("carrier") or {}).get("iata"),
        "flight_number": record.get("flightNumber"),
        "destination": (arr.get("airport") or {}).get("iata"),
        "block_minutes": minutes if isinstance(minutes, (int, float)) and minutes > 0 else None,  # 0 = impossible
        "miles": (record.get("distance") or {}).get("greatCircleMiles"),
        "passenger": (record.get("serviceType") or {}).get("iata") == "J",
        "international": ((arr.get("country") or {}).get("code") or "US") != "US",
        "origin": (dep.get("airport") or {}).get("iata"),
    }


class OAGClient:
    def __init__(self, http: httpx.Client | None = None, key: str | None = None, budget: int | None = None,
                 max_retries: int = 3):
        self.key = key or get_settings().require_oag_key()
        self.http = http or httpx.Client(timeout=60)
        self.budget = budget if budget is not None else load_config("scoring")["limits"]["oag_calls_per_question"]
        self.max_retries = max_retries
        self.calls = 0

    def _get(self, url: str, params: dict | None = None) -> dict:
        if urlparse(url).netloc != API_HOST:
            raise OAGError(f"refusing to send the OAG key to another host: {urlparse(url).netloc}")
        for attempt in range(self.max_retries + 1):
            if self.calls >= self.budget:
                raise BudgetExceeded(f"OAG call budget of {self.budget} reached")
            self.calls += 1
            try:
                r = self.http.get(url, params=params, headers={"Subscription-Key": self.key})
            except httpx.HTTPError as e:
                if attempt == self.max_retries:
                    raise OAGError(_safe(f"OAG request failed: {type(e).__name__}", self.key)) from None
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 401:
                raise OAGError("OAG rejected the key (401). Check OAG_API_KEY.")
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    raise OAGError(f"OAG kept failing (HTTP {r.status_code}) after {self.max_retries} retries")
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
            if r.status_code != 200:
                raise OAGError(_safe(f"OAG error HTTP {r.status_code}: {r.text[:200]}", self.key))
            return r.json()
        raise OAGError("unreachable")

    def departures(self, airport: str, date: str, use_cache: bool = True) -> FetchResult:
        """All operating departures from `airport` on `date` (YYYY-MM-DD)."""
        path = _cache_path(airport, date)
        if use_cache and path.exists():
            cached = json.loads(path.read_text())
            return FetchResult(cached["flights"], 0, True, cached["total_records"])
        params = {"DepartureDateTime": date, "DepartureAirport": airport, "CodeType": "IATA", "version": "v2"}
        first = self._get(BASE_URL, params)
        paging = first.get("paging") or {}
        pages = int(paging.get("totalPages") or 1)
        if pages > self.budget:
            raise BudgetExceeded(f"{airport} on {date} needs {pages} OAG calls; the budget is {self.budget}")
        records = list(first.get("data") or [])
        nxt = paging.get("next")
        while nxt:
            body = self._get(urljoin(BASE_URL, nxt))
            records += body.get("data") or []
            nxt = (body.get("paging") or {}).get("next")
        flights = [f for f in (operating_flight(r) for r in records) if f]
        total = int(paging.get("totalCount") or len(records))
        path.write_text(json.dumps({"flights": flights, "total_records": total,
                                    "fetched_records": len(records), "digest": hashlib.sha1(
                                        json.dumps(flights, sort_keys=True).encode()).hexdigest()}))
        return FetchResult(flights, self.calls, False, total)
