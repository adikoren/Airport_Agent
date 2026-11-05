"""OpenRouter client with mocked HTTP (respx): request shape, parsing, failures, key safety."""

import json

import httpx
import pytest
import respx

from airport_agent.llm.client import ClaudeUnavailable
from airport_agent.llm.openrouter import API_URL, OpenRouterLLM, strict_schema
from airport_agent.llm.schemas import ParsedRequest

KEY = "sk-or-test-key-123"


def llm():
    return OpenRouterLLM(key=KEY, model="anthropic/claude-opus-5.5")


def reply(content, annotations=None, finish="stop"):
    return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": {
        "role": "assistant", "content": content, "annotations": annotations or []}}]})


PARSED = {"intent": "rank", "airports": [], "region": "Texas", "metrics": [], "horizon_years": None, "weights": None,
          "long_haul_definition": None, "long_haul_threshold": None, "top_n": 3, "rank_by": "passengers",
          "order": "highest", "follow_up": False, "out_of_scope_parts": []}


def test_strict_schema_requires_every_field_and_allows_null():
    s = strict_schema(ParsedRequest)
    assert set(s["required"]) == set(s["properties"]) and s["additionalProperties"] is False
    assert "$ref" not in json.dumps(s) and "$defs" not in s
    assert any(t.get("type") == "null" for t in s["properties"]["region"]["anyOf"])


@respx.mock
def test_understand_sends_structured_request_and_parses_it():
    route = respx.post(API_URL).mock(return_value=reply(json.dumps(PARSED)))
    out = llm().understand([{"role": "user", "content": "3 busiest airports in Texas"}], "none")
    body = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["Authorization"] == f"Bearer {KEY}"
    assert body["model"] == "anthropic/claude-opus-5.5"
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"]["require_parameters"] is True and body["reasoning"]["effort"] == "low"
    assert body["messages"][0]["role"] == "system"
    assert out.rank_by == "passengers" and out.top_n == 3


@respx.mock
def test_answer_not_matching_the_structure_is_rejected():
    respx.post(API_URL).mock(return_value=reply('{"intent": "dance"}'))
    with pytest.raises(ClaudeUnavailable, match="structure"):
        llm().understand([{"role": "user", "content": "x"}], "none")


@respx.mock
@pytest.mark.parametrize("status,body,expect", [(402, '{"error":{"message":"Insufficient credits"}}', "too few credits"),
                                                (401, '{"error":{"message":"bad key"}}', "key was rejected"),
                                                (429, "{}", "rate limit"), (503, "{}", "provider is having problems")])
def test_failures_are_named_plainly_and_never_leak_the_key(status, body, expect):
    respx.post(API_URL).mock(return_value=httpx.Response(status, text=body.replace("bad key", f"bad key {KEY}")))
    with pytest.raises(ClaudeUnavailable) as e:
        llm().understand([{"role": "user", "content": "x"}], "none")
    assert expect in str(e.value) and KEY not in str(e.value)


@respx.mock
def test_refusal_is_treated_as_unavailable():
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                             "message": {"content": None, "refusal": "no"}}]}))
    with pytest.raises(ClaudeUnavailable):
        llm().understand([{"role": "user", "content": "x"}], "none")


@respx.mock
def test_web_lookup_uses_the_web_plugin_and_returns_cited_notes():
    ann = [{"type": "url_citation", "url_citation": {"url": "https://airport.example/facts", "title": "t"}}]
    route = respx.post(API_URL).mock(return_value=reply("- The airport has 115 gates [airport website]", ann))
    notes = llm().web_lookup("how many gates?", ["XXX"], ["the whole question"])
    body = json.loads(route.calls[0].request.content)
    assert body["plugins"][0]["id"] == "web"
    assert notes[0]["sources"] == [{"url": "https://airport.example/facts", "title": "t", "date": None}]


@respx.mock
def test_web_search_failure_is_raised_so_the_answer_can_say_it_did_not_run():
    respx.post(API_URL).mock(return_value=httpx.Response(503, text="{}"))
    with pytest.raises(ClaudeUnavailable):
        llm().web_lookup("q", [], [])


def test_missing_key_is_reported(monkeypatch):
    from airport_agent import settings
    monkeypatch.setattr(settings.Settings, "openrouter_api_key", None, raising=False)
    s = settings.Settings()
    s.openrouter_api_key = None
    monkeypatch.setattr("airport_agent.llm.openrouter.get_settings", lambda: s)
    with pytest.raises(ClaudeUnavailable, match="OPENROUTER_API_KEY"):
        OpenRouterLLM()


def test_provider_switch(monkeypatch):
    from airport_agent.graph import nodes
    from airport_agent.llm.client import Claude
    from airport_agent.settings import Settings
    for provider, expected in [("openrouter", OpenRouterLLM), ("anthropic", Claude)]:
        s = Settings()
        s.llm_provider = provider
        monkeypatch.setattr(nodes, "get_settings", lambda s=s: s)
        monkeypatch.delenv("AIRPORT_AGENT_CLAUDE", raising=False)
        monkeypatch.setattr(OpenRouterLLM, "__post_init__", lambda self: None)
        monkeypatch.setattr(Claude, "__post_init__", lambda self: None)
        nodes.set_claude(None)
        assert isinstance(nodes.claude(), expected)
    nodes.set_claude(None)
