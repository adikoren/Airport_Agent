"""Regression: one submit of the question form runs the question once. A rerun must not queue it again."""

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from streamlit.testing.v1 import AppTest

from airport_agent.graph import builder, nodes
from airport_agent.llm.fake import FakeClaude

APP = str(Path(__file__).resolve().parents[2] / "src/airport_agent/ui/app.py")


def test_one_submit_gives_one_turn_and_reruns_do_not_repeat_it(monkeypatch):
    monkeypatch.setattr(builder, "sqlite_checkpointer", lambda: MemorySaver())   # no file writes in the real cache
    nodes.set_claude(FakeClaude())
    try:
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.text_input(key="question_box").input("What is an IRR?")
        next(b for b in at.button if b.label == "Ask").click()
        at.run()
        for _ in range(3):                      # the page keeps rerunning; the question must not be queued again
            at.run()
        assert not at.exception
        assert len(at.session_state["turns"]) == 1
        assert at.session_state["turns"][0]["question"] == "What is an IRR?"
        assert "pending" not in at.session_state
    finally:
        nodes.set_claude(None)
