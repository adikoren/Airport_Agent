"""The LLM calls through OpenRouter (the default gateway; LLM_PROVIDER=openrouter).

Same model and the same five calls as the direct Anthropic client (llm/client.py):
understand, disambiguate, web_context, web_lookup, explain. The model never computes numbers here.

OpenRouter specifics (https://openrouter.ai/docs):
- POST /api/v1/chat/completions, `Authorization: Bearer <OPENROUTER_API_KEY>` (sent only to openrouter.ai).
- Structured output: `response_format = {type: json_schema, json_schema: {name, strict: true, schema}}`, with
  `provider.require_parameters = true` so only endpoints that support it are used. The answer is still
  validated with Pydantic here, because strict enforcement varies by provider.
- Web search: `plugins = [{id: "web", max_results}]`; citations in `message.annotations` (type url_citation).
- Reasoning depth: `reasoning = {effort}`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, ValidationError

from airport_agent.llm.client import (
    EFFORT,
    ClaudeUnavailable,
    _merge_user_turns,
    prompt,
)
from airport_agent.llm.schemas import Disambiguation, ParsedRequest
from airport_agent.settings import get_settings

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema → strict JSON schema: refs inlined, every property required, no extra properties.
    Optional fields stay nullable (anyOf [..., null]), so 'required' never forces a value."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            node = {k: walk(v) for k, v in node.items() if k not in ("default", "title")}
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            return node
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk(raw)


def api_problem_http(status: int, body: str) -> str:
    """A plain description of an OpenRouter failure (never the key or the prompt)."""
    low = body.lower()
    if status == 402 or "credit" in low:
        return "LLM API unavailable: the OpenRouter account has too few credits"
    if status == 401:
        return "LLM API unavailable: the OpenRouter API key was rejected"
    if status == 429:
        return "LLM API unavailable: rate limit reached, try again shortly"
    if status >= 500:
        return "LLM API unavailable: the provider is having problems, try again shortly"
    return f"LLM API unavailable: HTTP {status}"


@dataclass
class OpenRouterLLM:
    http: httpx.Client | None = None
    model: str | None = None
    key: str | None = field(default=None, repr=False)

    def __post_init__(self):
        s = get_settings()
        self.model = self.model or s.openrouter_model
        if self.key is None:
            if s.openrouter_api_key is None:
                raise ClaudeUnavailable("OPENROUTER_API_KEY is not set (add it to .env)")
            self.key = s.openrouter_api_key.get_secret_value()
        self.http = self.http or httpx.Client(timeout=180)

    # ---------------------------------------------------------------- transport

    def _post(self, body: dict) -> dict:
        try:
            r = self.http.post(API_URL, json={"model": self.model, **body},
                               headers={"Authorization": f"Bearer {self.key}",
                                        "X-Title": "Airport Investment Intelligence Agent"})
        except httpx.HTTPError as e:
            raise ClaudeUnavailable(f"LLM API unavailable: no connection ({type(e).__name__})") from None
        if r.status_code != 200:
            raise ClaudeUnavailable(api_problem_http(r.status_code, r.text[:500].replace(self.key, "***")))
        data = r.json()
        if data.get("error"):
            raise ClaudeUnavailable(api_problem_http(int(data["error"].get("code") or 500), str(data["error"])))
        return data

    @staticmethod
    def _message(data: dict) -> dict:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        if msg.get("refusal") or choice.get("finish_reason") in ("content_filter",):
            raise ClaudeUnavailable("the model declined to answer")
        return msg

    def _parse(self, system: str, messages: list[dict], schema: type[BaseModel], effort: str) -> BaseModel:
        data = self._post({
            "messages": [{"role": "system", "content": system}] + messages,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": schema.__name__, "strict": True,
                                                "schema": strict_schema(schema)}},
            "provider": {"require_parameters": True},
            "reasoning": {"effort": effort},
            "max_tokens": 4000,
        })
        content = self._message(data).get("content") or ""
        try:
            return schema.model_validate_json(content)
        except ValidationError:
            raise ClaudeUnavailable("the answer didn't match the required structure") from None

    def _search(self, system: str, user: str, effort: str, max_results: int = 5) -> list[dict]:
        """Web search through OpenRouter. Raises ClaudeUnavailable if the search could not run."""
        data = self._post({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                           "plugins": [{"id": "web", "max_results": max_results}],
                           "reasoning": {"effort": effort}, "max_tokens": 3000})
        msg = self._message(data)
        text = (msg.get("content") or "").strip()
        sources, seen = [], set()
        for a in msg.get("annotations") or []:
            c = a.get("url_citation") or {}
            if a.get("type") == "url_citation" and c.get("url") and c["url"] not in seen:
                seen.add(c["url"])
                sources.append({"url": c["url"], "title": c.get("title"), "date": None})
        return [{"text": text, "sources": sources[:6]}] if text else []

    # ---------------------------------------------------------------- the calls

    def understand(self, history: list[dict], context_summary: str) -> ParsedRequest:
        system = prompt("understand") + "\n\n## Conversation context\n" + context_summary
        return self._parse(system, _merge_user_turns(history), ParsedRequest, EFFORT["understand"])

    def disambiguate(self, history: list[dict], options: dict[str, list[dict]]) -> Disambiguation:
        msg = history + [{"role": "user", "content": "Candidates per mention (choose only from these codes, or null "
                                                     "to ask the user):\n" + json.dumps(options, indent=1)}]
        return self._parse(prompt("disambiguate"), _merge_user_turns(msg), Disambiguation, EFFORT["disambiguate"])

    def web_context(self, question: str, airports: list[str]) -> list[dict]:
        return self._search(prompt("web_context"), f"Question: {question}\nAirports: {', '.join(airports) or '(none named)'}",
                            EFFORT["web_context"], max_results=3)

    def web_lookup(self, question: str, airports: list[str], gaps: list[str]) -> list[dict]:
        user = (f"Question: {question}\nAirports: {', '.join(airports) or '(none named)'}\n"
                f"Missing from the agent's data: {'; '.join(gaps) or '(the whole question)'}")
        return self._search(prompt("web_lookup"), user, EFFORT["web_context"], max_results=5)

    def explain(self, history: list[dict], facts: dict, problems: list[str] | None = None) -> str:
        content = "FACTS (the only numbers you may use):\n" + json.dumps(facts, indent=1, default=str)
        if problems:
            content += "\n\nYour previous draft was rejected for these reasons. Fix all of them:\n- " + "\n- ".join(problems)
        messages = _merge_user_turns(history + [{"role": "user", "content": content}])
        data = self._post({"messages": [{"role": "system", "content": prompt("explain")}] + messages,
                           "reasoning": {"effort": EFFORT["explain"]}, "max_tokens": 6000})
        return (self._message(data).get("content") or "").strip()
