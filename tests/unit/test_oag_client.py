"""OAG client tests with mocked HTTP (respx). Uses the real response saved by the Phase 0 spike."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from airport_agent.tools import oag_client as oc

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "oag" / "flight_instances_BOS_2026-10-15.json"
KEY = "secret-test-key-123"


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(oc, "_cache_path", lambda a, d: tmp_path / f"{a}_{d}.json")
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)


def page(records, next_url=None, total_pages=1):
    return {"data": records, "paging": {"limit": 10, "totalCount": len(records), "totalPages": total_pages,
                                         **({"next": next_url} if next_url else {})}}


def real_records():
    return json.loads(FIX.read_text())["data"]


def test_codeshare_copies_are_dropped():
    flights = [f for f in (oc.operating_flight(r) for r in real_records()) if f]
    copies = [r for r in real_records() if r["codeshare"].get("operatingFlight")]
    assert len(flights) == len(real_records()) - len(copies) and copies


def test_zero_block_time_becomes_missing():
    r = dict(real_records()[2], elapsedTime=0)
    assert oc.operating_flight(r)["block_minutes"] is None


@respx.mock
def test_auth_header_params_and_paging():
    recs = real_records()
    route = respx.get("https://api.oag.com/flight-instances/").mock(side_effect=[
        httpx.Response(200, json=page(recs[:5], "https://api.oag.com/flight-instances?After=x", total_pages=2))])
    respx.get("https://api.oag.com/flight-instances?After=x").mock(return_value=httpx.Response(200, json=page(recs[5:])))
    c = oc.OAGClient(key=KEY, budget=10)
    res = c.departures("BOS", "2026-10-15", use_cache=False)
    req = route.calls[0].request
    assert req.headers["Subscription-Key"] == KEY
    assert req.url.params["DepartureAirport"] == "BOS" and req.url.params["version"] == "v2"
    assert res.calls == 2 and len(res.flights) == sum(1 for r in recs if not r["codeshare"].get("operatingFlight"))


@respx.mock
def test_relative_next_is_resolved_against_api_host():
    respx.get("https://api.oag.com/flight-instances/").mock(
        return_value=httpx.Response(200, json=page([], "/flight-instances?After=y", total_pages=2)))
    nxt = respx.get("https://api.oag.com/flight-instances?After=y").mock(return_value=httpx.Response(200, json=page([])))
    oc.OAGClient(key=KEY, budget=10).departures("BOS", "2026-10-15", use_cache=False)
    assert nxt.called


@respx.mock
def test_key_never_sent_to_another_host():
    respx.get("https://api.oag.com/flight-instances/").mock(
        return_value=httpx.Response(200, json=page([], "https://evil.example.com/steal", total_pages=2)))
    with pytest.raises(oc.OAGError, match="another host"):
        oc.OAGClient(key=KEY, budget=10).departures("BOS", "2026-10-15", use_cache=False)


@respx.mock
def test_429_is_retried_then_succeeds():
    respx.get("https://api.oag.com/flight-instances/").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200, json=page([]))])
    res = oc.OAGClient(key=KEY, budget=10).departures("BOS", "2026-10-15", use_cache=False)
    assert res.calls == 2


@respx.mock
def test_errors_do_not_leak_the_key():
    respx.get("https://api.oag.com/flight-instances/").mock(return_value=httpx.Response(400, text=f"bad {KEY}"))
    with pytest.raises(oc.OAGError) as e:
        oc.OAGClient(key=KEY, budget=10).departures("BOS", "2026-10-15", use_cache=False)
    assert KEY not in str(e.value)


@respx.mock
def test_day_too_big_for_budget_stops_before_paging():
    respx.get("https://api.oag.com/flight-instances/").mock(return_value=httpx.Response(200, json=page([], "/x", total_pages=177)))
    c = oc.OAGClient(key=KEY, budget=50)
    with pytest.raises(oc.BudgetExceeded):
        c.departures("BOS", "2026-10-15", use_cache=False)
    assert c.calls == 1


@respx.mock
def test_cache_reuse_makes_no_calls():
    respx.get("https://api.oag.com/flight-instances/").mock(return_value=httpx.Response(200, json=page(real_records()[:3])))
    oc.OAGClient(key=KEY, budget=10).departures("BOS", "2026-10-15")
    again = oc.OAGClient(key=KEY, budget=10).departures("BOS", "2026-10-15")
    assert again.from_cache and again.calls == 0
