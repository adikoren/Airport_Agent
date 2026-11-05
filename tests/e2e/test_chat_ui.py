"""End-to-end tests: the real Streamlit app in a real browser (Playwright).

Offline tests use the scripted Claude (AIRPORT_AGENT_CLAUDE=fake), so they check the UI and the whole
agent flow — chat, newest-first order, per-answer execution details, follow-ups, clarification, scope reply, reset —
for free. The `live` test uses the real Claude and runs only with RUN_LIVE_E2E=1.

Layout under test: the input is at the top; the newest question-answer pair is shown first; each answer has its own
'How this answer was built' expander directly below it.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

ANSWER_TIMEOUT = 60_000          # ms; offline answers take a few seconds, live ones up to ~60 s
# Questions the tests send (the home page has no example buttons).
EXAMPLES = ["Compare LA and Santa Ana airport congestion levels.",
            "What is the percentage of long haul flights out of Anchorage airport?",
            "What is the unmet flight demand in SFO airport and why?"]
RANKING_QUESTION = "Which airports in New England are strong candidates for terminal expansion?"
TRACE_TITLE = "How this answer was built"


def open_app(page: Page, url: str) -> None:
    page.goto(url)
    expect(page.get_by_text("Airport Investment Intelligence Agent")).to_be_visible(timeout=30_000)


def wait_until_answered(page: Page, expected_messages: int, question: str) -> None:
    """Wait until the answer to `question` is shown: the newest pair starts with the question, its details exist,
    no spinner is left, and no stale elements remain."""
    expect(page.get_by_test_id("stChatMessage")).to_have_count(expected_messages, timeout=ANSWER_TIMEOUT)
    expect(page.get_by_test_id("stChatMessage").first).to_contain_text(question, timeout=ANSWER_TIMEOUT)
    expect(page.locator("summary", has_text=TRACE_TITLE)).to_have_count(expected_messages // 2,
                                                                       timeout=ANSWER_TIMEOUT)
    expect(page.get_by_test_id("stSpinner")).to_have_count(0, timeout=ANSWER_TIMEOUT)
    # after a run Streamlit briefly keeps the previous elements, marked stale: wait until they are replaced
    expect(page.locator('[data-stale="true"]')).to_have_count(0, timeout=ANSWER_TIMEOUT)


def ask(page: Page, text: str) -> None:
    """Type a question in the chat box, send it, and wait for the agent's answer."""
    before = page.get_by_test_id("stChatMessage").count()
    box = page.get_by_role("textbox")
    box.fill(text)
    box.press("Enter")
    wait_until_answered(page, before + 2, text)


def click_example(page: Page, question: str) -> None:
    """Send a question by typing it."""
    ask(page, question)


def last_answer(page: Page):
    """The newest answer: it is the second message of the newest pair, shown at the top."""
    return page.get_by_test_id("stChatMessage").nth(1)


def trace(page: Page, index: int = 0):
    """The 'How this answer was built' expander of the answer at `index` (0 = newest), opened. Returns its text."""
    summary = page.locator("summary", has_text=TRACE_TITLE).nth(index)
    if not summary.evaluate("e => e.parentElement.open"):
        summary.click()
    return summary.locator("xpath=..").inner_text()


def nodes_ran(page: Page, index: int = 0) -> str:
    """The 'Nodes executed' line of that answer's details."""
    text = trace(page, index)
    match = re.search(r"Nodes executed:\s*(.+)", text)
    return match.group(1) if match else ""


# ---------------------------------------------------------------- offline (scripted Claude)

def test_home_page_has_input_at_the_top_and_no_example_buttons(page: Page, app_url):
    open_app(page, app_url)
    for q in EXAMPLES + [RANKING_QUESTION]:
        expect(page.get_by_role("button", name=q)).to_have_count(0)
    expect(page.get_by_role("textbox")).to_be_visible()
    expect(page.get_by_test_id("stChatMessage")).to_have_count(0)


def test_ranking_answer_carries_its_execution_details(page: Page, app_url):
    open_app(page, app_url)
    ask(page, RANKING_QUESTION)
    answer = last_answer(page)
    expect(answer).to_contain_text("Renovation candidates in New England")
    expect(answer).to_contain_text("Assumptions & uncertainty")
    expect(answer).to_contain_text("not investment advice")
    expect(answer.locator("table")).to_be_visible()
    # the full data path ran, and the details say so
    ran = nodes_ran(page)
    for node in ["understand_request", "resolve_airports", "get_data", "compute_metrics", "score_and_rank",
                 "prepare_results", "explain", "output_guard", "respond"]:
        assert node in ran
    assert "Question: " + RANKING_QUESTION in trace(page)


def test_follow_ups_reuse_results_and_rescore(page: Page, app_url):
    open_app(page, app_url)
    ask(page, RANKING_QUESTION)
    first_answer = last_answer(page).inner_text()
    top = re.search(r"\|\s*1\s*\|\s*[^|]*\((\w{3})\)", first_answer) or re.search(r"\((\w{3})\)", first_answer)

    ask(page, "Why is the first one first?")
    ran = nodes_ran(page)
    assert "get_data" not in ran and "compute_metrics" not in ran and "score_and_rank" not in ran
    assert "prepare_results" in ran
    assert "Stored results: Reused" in trace(page)              # the details say the stored results were reused
    if top:
        expect(last_answer(page)).to_contain_text(top.group(1))

    ask(page, "And over a 10-year horizon?")
    ran = nodes_ran(page)
    assert "score_and_rank" in ran and "get_data" not in ran and "compute_metrics" not in ran
    expect(last_answer(page)).to_contain_text(re.compile(r"10[- ]year"))


def test_newest_pair_first_and_each_answer_keeps_its_own_details(page: Page, app_url):
    open_app(page, app_url)
    ask(page, RANKING_QUESTION)
    ask(page, "How do you score?")
    # the newest pair is at the top: its question comes first, then its answer
    expect(page.get_by_test_id("stChatMessage").nth(0)).to_contain_text("How do you score?")
    expect(page.get_by_test_id("stChatMessage").nth(1)).to_contain_text("RPS = 100")
    expect(page.get_by_test_id("stChatMessage").nth(2)).to_contain_text(RANKING_QUESTION)
    # each details panel belongs to its own answer and holds its own question and nodes
    assert "Question: How do you score?" in trace(page, 0)
    assert "get_data" not in nodes_ran(page, 0)                  # the methodology answer fetched nothing
    older = trace(page, 1)
    assert f"Question: {RANKING_QUESTION}" in older and "get_data" in nodes_ran(page, 1)


def test_city_with_two_airports_asks_the_user(page: Page, app_url):
    open_app(page, app_url)
    ask(page, "How congested is Portland?")
    answer = last_answer(page)
    expect(answer).to_contain_text("Which one do you mean")
    expect(answer).to_contain_text("(PDX)")
    expect(answer).to_contain_text("(PWM)")
    assert "explain" not in nodes_ran(page)
    assert "clarification" in trace(page)


def test_metro_name_resolved_from_context(page: Page, app_url):
    open_app(page, app_url)
    click_example(page, EXAMPLES[0])
    answer = last_answer(page)
    expect(answer).to_contain_text("Constraint comparison", timeout=ANSWER_TIMEOUT)
    expect(answer).to_contain_text("(LAX)")
    expect(answer).to_contain_text("(SNA)")
    assert "disambiguate_airports" in nodes_ran(page)


def test_long_haul_details_show_the_oag_cache(page: Page, app_url):
    open_app(page, app_url)
    click_example(page, EXAMPLES[1])
    answer = last_answer(page)
    expect(answer).to_contain_text("Long-haul share", timeout=ANSWER_TIMEOUT)
    expect(answer).to_contain_text("By block time")
    expect(answer).to_contain_text("By distance")
    assert "from the local cache" in trace(page)                  # served from the OAG cache, shown as such


def test_why_question_adds_web_background(page: Page, app_url):
    open_app(page, app_url)
    click_example(page, EXAMPLES[2])
    expect(last_answer(page)).to_contain_text("(SFO)", timeout=ANSWER_TIMEOUT)
    assert "web_context" in nodes_ran(page)
    expect(last_answer(page).get_by_text("Background from web sources")).to_be_visible()
    assert "Web search" in trace(page)


def test_out_of_scope_gets_the_fixed_reply(page: Page, app_url):
    open_app(page, app_url)
    ask(page, "What is the IRR of a new terminal?")
    expect(last_answer(page)).to_contain_text("I can't do that part")
    ran = nodes_ran(page)
    assert "explain" not in ran and "get_data" not in ran
    assert "scope reply" in trace(page)


def test_methodology_answer_needs_no_data(page: Page, app_url):
    open_app(page, app_url)
    ask(page, "How do you score?")
    expect(last_answer(page)).to_contain_text("RPS = 100")
    assert "get_data" not in nodes_ran(page)


def test_chat_box_is_disabled_while_the_agent_works(page: Page, app_url):
    open_app(page, app_url)
    box = page.get_by_role("textbox")
    box.fill(RANKING_QUESTION)
    box.press("Enter")
    expect(box).to_be_disabled(timeout=10_000)
    wait_until_answered(page, 2, RANKING_QUESTION)
    expect(page.get_by_role("textbox")).to_be_enabled()


def test_new_conversation_resets_the_chat(page: Page, app_url):
    open_app(page, app_url)
    ask(page, "How do you score?")
    page.get_by_role("button", name="New conversation").click()
    expect(page.get_by_test_id("stChatMessage")).to_have_count(0, timeout=10_000)
    expect(page.get_by_role("textbox")).to_be_visible()


# ---------------------------------------------------------------- live (real Claude, opt-in)

@pytest.mark.live
def test_live_ranking_and_follow_up(page: Page, live_app_url):
    open_app(page, live_app_url)
    ask(page, RANKING_QUESTION)
    expect(page.get_by_test_id("stSpinner")).to_have_count(0, timeout=180_000)
    answer = last_answer(page)
    expect(answer).to_contain_text("Assumptions & uncertainty", timeout=120_000)
    expect(answer).to_contain_text("BOS")
    assert "fixed-format answer" not in answer.inner_text()      # Claude's draft passed the number check
    ask(page, "Why is the first one first?")
    assert "get_data" not in nodes_ran(page)


def test_question_beyond_the_data_shows_the_labeled_web_answer(page: Page, app_url):
    open_app(page, app_url)
    ask(page, "How many gates does Denver have?")
    answer = last_answer(page)
    expect(answer).to_contain_text("From web sources (not from the agent's data)")
    link = answer.get_by_role("link", name="Offline test source")          # the source is shown as a link
    expect(link).to_have_attribute("href", "https://example.org/offline-test")
    assert "web_context" in nodes_ran(page)
