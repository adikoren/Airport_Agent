"""Streamlit chat for the Airport Investment Intelligence Agent.

Run:  .venv/bin/streamlit run src/airport_agent/ui/app.py

Each turn is stored as a snapshot (question, answer, execution details, facts). The input sits at the top and the
newest question-answer pair is shown first. Display order is reversed only here; the graph and its memory keep the
chronological order.
"""

from __future__ import annotations

import faulthandler
import signal
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

# Diagnostics: `kill -USR1 <pid>` writes the Python stack of every thread to the server log.
faulthandler.register(signal.SIGUSR1, all_threads=True)

from airport_agent.graph.builder import build, sqlite_checkpointer
from airport_agent.ui.trace import build_trace

st.set_page_config(page_title="Airport Investment Agent", page_icon="✈️", layout="centered")


@st.cache_resource
def graph():
    return build(sqlite_checkpointer())


def run_turn(text: str) -> dict:
    """Run one question through the graph and return the stored snapshot for it."""
    cfg = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 25}
    updates: list[tuple[str, dict]] = []
    try:
        for update in graph().stream({"messages": [{"role": "user", "content": text}]}, config=cfg,
                                     stream_mode="updates"):
            for node, writes in update.items():
                updates.append((node, writes or {}))
        final = graph().get_state(cfg).values
        answer = final["messages"][-1]["content"]
        trace = build_trace(text, updates, final)
    except Exception as e:  # noqa: BLE001 — never leave the chat box disabled after an unexpected error
        final = {}
        answer = f"Something went wrong while answering ({type(e).__name__}). Please try again."
        trace = {"question": text, "steps": [n for n, _ in updates], "outcome": "unexpected error",
                 "errors": [type(e).__name__]}
    return {"question": text, "answer": answer, "trace": trace, "facts": final.get("facts") or {}}


def render_trace(turn: dict) -> None:
    """The execution details of one answer. Only the fields that apply to this turn are shown."""
    tr = turn["trace"]
    st.markdown(f"**Question:** {tr['question']}")
    st.markdown("**Nodes executed:** " + " → ".join(f"`{s}`" for s in tr["steps"]))
    if tr.get("intent"):
        st.markdown(f"**Request type:** {tr['intent']}")
    if tr.get("llm_calls"):
        st.markdown(f"**Claude calls ({len(tr['llm_calls'])}):**")
        st.markdown("\n".join(f"- {c}" for c in tr["llm_calls"]))
    if tr.get("api_calls"):
        st.markdown("**External API calls:**\n" + "\n".join(f"- {c}" for c in tr["api_calls"]))
    if tr.get("local_data"):
        st.markdown(f"**Data:** {tr['local_data']}")
    if tr.get("reused"):
        st.markdown(f"**Stored results:** {tr['reused']}")
    if tr.get("web"):
        st.markdown(f"**Web search:** {tr['web']}")
    if tr.get("rejections"):
        st.markdown("**Number check rejected the draft:**")
        for r in tr["rejections"]:
            st.markdown(f"- Attempt {r['attempt']}: {'; '.join(r['problems'])} → {r['action']}")
    st.markdown(f"**Final answer:** {tr['outcome']}")
    if tr.get("errors"):
        st.markdown("**Errors:**\n" + "\n".join(f"- {e}" for e in tr["errors"]))
    if turn.get("facts") and turn["facts"].get("answer_mode") == "computed":
        with st.expander("Technical detail: facts bundle (the only numbers the model may use)"):
            st.json(turn["facts"], expanded=False)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.turns = []          # chronological; displayed newest first

title_col, reset_col = st.columns([5, 2])
with title_col:
    st.title("Airport Investment Intelligence Agent")
with reset_col:
    st.write("")
    if st.button("New conversation"):
        st.session_state.clear()
        st.rerun()

# The input is at the top (st.chat_input would be docked to the bottom of the page, so a form is used instead).
# Two-step submit: store the question and rerun, so the box is shown disabled while the agent works. Otherwise a
# second question typed meanwhile would interrupt the running answer.
busy = "pending" in st.session_state


def queue_question() -> None:
    """Runs once, when the form is submitted. The text is read here, not on later reruns, so it is queued only once."""
    text = (st.session_state.get("question_box") or "").strip()
    if text and "pending" not in st.session_state:
        st.session_state.pending = text


with st.form("ask", clear_on_submit=True, border=False):
    st.text_input("Question", key="question_box",
                  placeholder="Working on your question…" if busy else "Ask about US airports…",
                  disabled=busy, label_visibility="collapsed")
    st.form_submit_button("Ask", disabled=busy, on_click=queue_question)

if "pending" in st.session_state:
    text = st.session_state.pending
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"), st.spinner("Working…"):
        turn = run_turn(text)
    st.session_state.turns.append(turn)
    st.session_state.pop("pending", None)
    st.rerun()

for turn in reversed(st.session_state.turns):       # newest pair first; each answer carries its own details
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
    with st.expander("How this answer was built"):
        render_trace(turn)
