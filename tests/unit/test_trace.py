"""Per-turn execution details: built only from what happened in that turn, and shown only where they apply."""

from airport_agent.ui.trace import build_trace


def guard(answer, problems):
    return ("output_guard", {"answer": answer, "guard_problems": problems})


def test_retries_are_listed_in_order_and_the_rejection_reasons_are_kept():
    updates = [("understand_request", {"request": {"intent": "explain_why"}}),
               ("prepare_results", {"facts": {}}),
               ("explain", {"draft_answer": "a"}), guard(None, ["numbers not in the data: 9"]),
               ("explain", {"draft_answer": "b"}), guard("the good answer", [])]
    tr = build_trace("why?", updates, {"request": {"intent": "explain_why"}, "counters": {"claude_calls": 3}})
    assert tr["steps"] == ["understand_request", "prepare_results", "explain", "output_guard",
                           "explain (retry 1)", "output_guard (retry 1)"]
    assert tr["outcome"] == "rewritten after the number check"
    assert tr["rejections"] == [{"attempt": 1, "problems": ["numbers not in the data: 9"],
                                 "action": "answer rewritten"}]
    assert tr["llm_calls"] == ["Understand the question", "Write the answer",
                               "Rewrite the answer after the number check"]


def test_a_fixed_format_fallback_is_named_as_such_with_its_reasons():
    updates = [("explain", {}), guard(None, ["missing the section"]), ("explain", {}),
               guard("the fixed-format answer", ["missing the section"])]
    tr = build_trace("q", updates, {"request": {"intent": "rank"}, "counters": {"claude_calls": 3}})
    assert tr["outcome"] == "fixed-format fallback"
    assert tr["rejections"][-1]["action"] == "fixed-format fallback used"


def test_stored_results_are_reported_only_for_a_reuse_turn():
    reuse = [("understand_request", {}), ("prepare_results", {}), ("explain", {}), guard("x", [])]
    tr = build_trace("why?", reuse, {"request": {"intent": "explain_why"}, "counters": {"claude_calls": 2}})
    assert tr["reused"].startswith("Reused the results of the previous answer")
    fresh = [("understand_request", {}), ("resolve_airports", {}), ("get_data", {}), ("compute_metrics", {}),
             ("score_and_rank", {}), ("prepare_results", {}), ("explain", {}), guard("x", [])]
    tr2 = build_trace("rank", fresh, {"request": {"intent": "rank"}, "counters": {"claude_calls": 2},
                                      "raw_data": {}})
    assert "reused" not in tr2 and tr2["local_data"].startswith("Public data read from local files")


def test_a_parameter_change_is_reported_as_a_rescore():
    updates = [("understand_request", {}), ("score_and_rank", {}), ("prepare_results", {}), ("explain", {}),
               guard("x", [])]
    tr = build_trace("10-year horizon", updates, {"request": {"intent": "rank"}, "counters": {"claude_calls": 2}})
    assert tr["reused"].startswith("Re-scored the stored measures")


def test_a_conceptual_answer_shows_no_data_api_or_web_lines():
    updates = [("understand_request", {}), ("prepare_results", {}), ("explain", {}), guard("x", [])]
    tr = build_trace("what is an IRR?", updates, {"request": {"intent": "conceptual"}, "counters": {"claude_calls": 2}})
    for key in ("api_calls", "local_data", "reused", "web", "rejections", "errors"):
        assert key not in tr
    assert tr["outcome"] == "answer"


def test_web_research_reports_its_status_and_a_failure_reason():
    updates = [("understand_request", {}), ("resolve_airports", {}), ("prepare_results", {}),
               ("web_context", {}), ("explain", {}), guard("x", [])]
    ok = build_trace("plans?", updates, {"request": {"intent": "research"}, "web_status": "results",
                                         "counters": {"claude_calls": 3}})
    assert ok["web"] == "Web search ran and returned sources"
    failed = build_trace("plans?", updates, {"request": {"intent": "research"}, "web_status": "failed",
                                             "web_error": "credit balance is too low", "counters": {"claude_calls": 3}})
    assert "did not run (credit balance is too low)" in failed["web"]


def test_oag_calls_show_cache_use_and_a_bts_fallback_is_named():
    updates = [("understand_request", {}), ("get_data", {}), ("prepare_results", {}), ("explain", {}), guard("x", [])]
    final = {"request": {"intent": "metric_lookup"}, "counters": {"claude_calls": 2, "api_calls": 0},
             "raw_data": {"long_haul": {"ANC": {"source": "OAG", "from_cache": True, "calls": 0},
                                        "HNL": {"source": "BTS", "fallback_reason": "over the call budget"}}}}
    tr = build_trace("long haul?", updates, final)
    assert "OAG schedule for ANC: 0 call(s), from the local cache" in tr["api_calls"]
    assert any("OAG not used for HNL (over the call budget)" in line for line in tr["api_calls"])


def test_clarification_errors_and_scope_replies_have_their_own_outcome():
    clar = build_trace("Portland?", [("understand_request", {}), ("respond", {})],
                       {"request": {"intent": "rank"}, "clarification": {"text": "Which Portland?"},
                        "counters": {"claude_calls": 1}})
    assert clar["outcome"] == "clarification"
    err = build_trace("q", [("understand_request", {"errors": ["understand_request: LLM API unavailable"]})],
                      {"request": None, "errors": ["understand_request: LLM API unavailable"], "counters": {}})
    assert err["outcome"] == "error" and err["errors"]
    scope = build_trace("weather?", [("understand_request", {}), ("respond", {})],
                        {"request": {"intent": "out_of_scope"}, "counters": {"claude_calls": 1}})
    assert scope["outcome"] == "scope reply"


def test_each_turn_keeps_its_own_trace():
    first = build_trace("q1", [("understand_request", {}), ("explain", {}), guard("a", [])],
                        {"request": {"intent": "rank"}, "counters": {"claude_calls": 2}})
    second = build_trace("q2", [("understand_request", {}), ("respond", {})],
                         {"request": {"intent": "out_of_scope"}, "counters": {"claude_calls": 1}})
    assert first["question"] == "q1" and first["outcome"] == "answer"
    assert second["question"] == "q2" and second["outcome"] == "scope reply"
    assert "Write the answer" in first["llm_calls"]
