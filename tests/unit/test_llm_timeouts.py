"""A model call that stalls must fail with a clear message within the configured time, never hang the chat."""

import anthropic
import httpx
import pydantic

from airport_agent.llm import client as C


def test_the_anthropic_client_gets_an_explicit_timeout_and_no_sdk_retries(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class Settings:
        anthropic_api_key = pydantic.SecretStr("sk-ant-test-not-real")
        claude_model = "claude-haiku-5-5"

    monkeypatch.setattr(C, "get_settings", lambda: Settings())
    monkeypatch.setattr(C.anthropic, "Anthropic", Recorder)
    C.Claude()
    assert seen["timeout"] == C.LLM_TIMEOUT_SECONDS == 60.0
    assert seen["max_retries"] == C.LLM_SDK_RETRIES == 0


def test_a_timeout_is_reported_as_a_timeout_not_as_no_connection():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    message = C.api_problem(anthropic.APITimeoutError(request=request))
    assert message.startswith("LLM API unavailable: no answer within 60 seconds")
    assert "no connection" not in message


def test_a_connection_error_is_still_reported_as_no_connection():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    assert C.api_problem(anthropic.APIConnectionError(request=request)) == "LLM API unavailable: no connection"


def test_a_timed_out_model_call_gives_a_clear_reply_and_the_turn_still_finishes():
    """A stalled model call must end the turn with a plain message, not leave it running."""
    import uuid

    from langgraph.checkpoint.memory import MemorySaver

    from airport_agent.graph import nodes
    from airport_agent.graph.builder import build

    class TimingOut:
        def understand(self, history, ctx):
            raise C.ClaudeUnavailable("LLM API unavailable: no answer within 60 seconds, try again")

    nodes.set_claude(TimingOut())
    try:
        graph = build(MemorySaver())
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}
        graph.invoke({"messages": [{"role": "user", "content": "What is an IRR?"}]}, config=cfg)
        text = graph.get_state(cfg).values["messages"][-1]["content"]
    finally:
        nodes.set_claude(None)
    assert "no answer within 60 seconds" in text and "fine" in text.lower()
