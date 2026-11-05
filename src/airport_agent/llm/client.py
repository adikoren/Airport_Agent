"""The four Claude calls (ARCHITECTURE §7). Model and effort per call come from config.

Claude never computes numbers here: it parses requests, picks among code-supplied candidates,
gathers labeled web context, and writes the answer from the facts bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anthropic
from pydantic import BaseModel

from airport_agent.llm.schemas import Disambiguation, ParsedRequest
from airport_agent.settings import get_settings

PROMPTS = Path(__file__).parent / "prompts"
EFFORT = {"understand": "low", "disambiguate": "low", "web_context": "low", "explain": "medium"}


WEB_PROMPTS = {"web_context", "web_lookup"}


def prompt(name: str) -> str:
    """A system prompt. Web prompts share one provenance file, so the rules cannot drift apart."""
    text = (PROMPTS / f"{name}.md").read_text()
    if name in WEB_PROMPTS:
        text += "\n\n" + (PROMPTS / "_provenance.md").read_text()
    return text


class ClaudeUnavailable(RuntimeError):
    """No usable answer from Claude (no key, API error, refusal). Callers send a fixed reply instead."""


@dataclass
class Claude:
    client: anthropic.Anthropic | None = None
    model: str | None = None

    def __post_init__(self):
        s = get_settings()
        self.model = self.model or s.claude_model
        if self.client is None:
            if s.anthropic_api_key is None:
                raise ClaudeUnavailable("ANTHROPIC_API_KEY is not set (add it to .env)")
            # base_url pinned, so an ANTHROPIC_BASE_URL left in the shell can't redirect the app's calls
            self.client = anthropic.Anthropic(api_key=s.anthropic_api_key.get_secret_value(),
                                              base_url="https://api.anthropic.com")

    def _parse(self, system: str, messages: list[dict], schema: type[BaseModel], effort: str) -> BaseModel:
        try:
            r = self.client.messages.parse(
                model=self.model, max_tokens=4000, system=[{"type": "text", "text": system,
                                                            "cache_control": {"type": "ephemeral"}}],
                messages=messages, output_format=schema, output_config={"effort": effort})
        except anthropic.APIError as e:
            raise ClaudeUnavailable(api_problem(e)) from None
        if r.stop_reason == "refusal" or r.parsed_output is None:
            raise ClaudeUnavailable(f"no structured answer (stop reason: {r.stop_reason})")
        return r.parsed_output

    # 1. understand_request
    def understand(self, history: list[dict], context_summary: str) -> ParsedRequest:
        system = prompt("understand") + "\n\n## Conversation context\n" + context_summary
        return self._parse(system, history, ParsedRequest, EFFORT["understand"])

    # 2. disambiguate_airports
    def disambiguate(self, history: list[dict], options: dict[str, list[dict]]) -> Disambiguation:
        msg = history + [{"role": "user", "content":
                          "Candidates per mention (choose only from these codes, or null to ask the user):\n"
                          + json.dumps(options, indent=1)}]
        return self._parse(prompt("disambiguate"), _merge_user_turns(msg), Disambiguation, EFFORT["disambiguate"])

    # 3. web research: background for a "why" question, or external facts (sourced, never scored)
    def web_context(self, question: str, airports: list[str]) -> list[dict]:
        return self._web(prompt("web_context"), f"Question: {question}\nAirports: {', '.join(airports) or '(none named)'}",
                         EFFORT["web_context"])

    def web_lookup(self, question: str, airports: list[str], gaps: list[str]) -> list[dict]:
        user = (f"Question: {question}\nAirports: {', '.join(airports) or '(none named)'}\n"
                f"Missing from the agent's data: {'; '.join(gaps) or '(the whole question)'}")
        return self._web(prompt("web_lookup"), user, EFFORT["web_context"])

    def _web(self, system: str, user: str, effort: str) -> list[dict]:
        """Run the search. Raises ClaudeUnavailable if the search could not run; returns [] if it ran but found nothing."""
        messages = [{"role": "user", "content": user}]
        try:
            for _ in range(3):                           # a long search can pause; resume it, up to 3 times
                r = self.client.messages.create(model=self.model, max_tokens=3000, system=system,
                                                tools=[web_search_tool(self.model)], messages=messages,
                                                output_config={"effort": effort})
                if r.stop_reason != "pause_turn":
                    break
                messages = messages + [{"role": "assistant", "content": r.content}]
        except anthropic.APIError as e:
            raise ClaudeUnavailable(api_problem(e)) from None
        if r.stop_reason == "refusal":
            return []
        return notes_from_blocks(r.content)

    # 4. explain
    def explain(self, history: list[dict], facts: dict, problems: list[str] | None = None) -> str:
        content = "FACTS (the only numbers you may use):\n" + json.dumps(facts, indent=1, default=str)
        if problems:
            content += "\n\nYour previous draft was rejected for these reasons. Fix all of them:\n- " + "\n- ".join(problems)
        messages = _merge_user_turns(history + [{"role": "user", "content": content}])
        kwargs = dict(model=self.model, max_tokens=6000,
                      system=[{"type": "text", "text": prompt("explain"), "cache_control": {"type": "ephemeral"}}],
                      messages=messages, output_config={"effort": EFFORT["explain"]})
        try:
            try:  # server-side refusal fallback (beta); plain call if the beta isn't available
                r = self.client.beta.messages.create(**kwargs, betas=["server-side-fallback-2026-07-01"],
                                                     fallbacks="default")
            except (anthropic.BadRequestError, TypeError):
                r = self.client.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise ClaudeUnavailable(api_problem(e)) from None
        if r.stop_reason == "refusal":
            raise ClaudeUnavailable("Claude declined to answer")
        return "".join(b.text for b in r.content if b.type == "text").strip()


def notes_from_blocks(blocks) -> list[dict]:
    """Text and its sources from a response: cited pages from search results and from text citations."""
    text_parts, sources = [], []
    for b in blocks:
        if b.type == "web_search_tool_result" and isinstance(b.content, list):
            for x in b.content:
                if getattr(x, "url", None):
                    sources.append({"url": x.url, "title": getattr(x, "title", None), "date": getattr(x, "page_age", None)})
        if b.type == "text" and b.text.strip():
            text_parts.append(b.text.strip())
            for c in getattr(b, "citations", None) or []:
                if getattr(c, "url", None):
                    sources.append({"url": c.url, "title": getattr(c, "title", None), "date": None})
    text = "\n".join(text_parts).strip()
    if not text:
        return []
    seen, unique = set(), []
    for src in sources:
        if src["url"] not in seen:
            seen.add(src["url"])
            unique.append(src)
    return [{"text": text, "sources": unique[:6]}]


def web_search_tool(model: str) -> dict:
    """Haiku keeps the basic search tool; Opus, Sonnet and later models use the filtered version."""
    kind = "web_search_20250305" if "haiku" in model else "web_search_20260209"
    return {"type": kind, "name": "web_search", "max_uses": 3}


def api_problem(e: anthropic.APIError) -> str:
    """A plain description of an API failure, for the user and the audit log (never the key or the prompt)."""
    msg = str(getattr(e, "message", "") or e).lower()
    if "credit balance" in msg:
        return "LLM API unavailable: the Anthropic account's credit balance is too low"
    if isinstance(e, anthropic.AuthenticationError):
        return "LLM API unavailable: the Anthropic API key was rejected"
    if isinstance(e, anthropic.RateLimitError):
        return "LLM API unavailable: rate limit reached, try again shortly"
    if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return "LLM API unavailable: no connection"
    return f"LLM API unavailable: {type(e).__name__}"


def _merge_user_turns(messages: list[dict]) -> list[dict]:
    """The API needs alternating roles; merge consecutive user turns."""
    out: list[dict] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"] == "user":
            out[-1] = {"role": "user", "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(dict(m))
    return out
