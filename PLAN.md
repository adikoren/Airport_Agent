# Airport Investment Intelligence Agent — Build Plan

> Companion to [ARCHITECTURE.md](ARCHITECTURE.md). Time budget: **one day**. Most of the plan is built; the status table below says what is built, what is verified, and what is planned.
> Rule for the whole build: **every deterministic piece is a tool with tests before it is wired into the graph.**

**Build status (2026-11-05)**

| Phase | Status |
|---|---|
| 0. OAG spike | ✅ Done: free tier found (10 records/call) → OAG for long haul only (D2), [docs/oag_notes.md](docs/oag_notes.md) |
| 1. Data foundation | ✅ Done: BTS aggregates, FAA TAF 2025, airport reference (359/359 airports matched), signal reference. Not done: TAF error from older editions (cut-list item 3) and the 2018 backtest |
| 2. Tools | ✅ Done: store, checks, metrics, scoring, resolver, OAG client, facts, output guard, CLI |
| 3. LangGraph | ✅ Done: 11 nodes, routing, scope rules, SQLite memory, audit log. **Verified live on Claude Haiku 5.5 before the answer-mode rewrite:** scope evaluation 17/17. 131 offline tests pass (unit, graph, held-out) |
| 4. Chat interface | ✅ Running (Streamlit); the chat box is disabled while an answer is being written. **End-to-end (Playwright):** 12 offline tests pass; 1 live test skipped (needs `RUN_LIVE_E2E=1`). Voice not started |
| 5. Evaluation and docs | README done. Scope evaluation saved (17 + 10 unseen phrasings, the unseen set run before the rewrite: 25/27). Design document: [DESIGN.md](DESIGN.md) updated in the answer-mode round |
| 5d. Answer modes and provenance (round of 2026-11-05) | ✅ **Implemented, tested offline:** answer modes (conceptual, research, computed, methodology, explanation of earlier results, out of scope); web results carry source and date and a status (results / no reliable result / failed); output guard by mode; prompts rewritten and generalised; methodology corrections in text and config. ⚠️ **Not verified live:** the rewritten prompts and new modes have not been run against the live model, and the web search tool has not been run live on Haiku 5.5. See §5d |

**How to read this document**

| Section | Answers |
|---|---|
| 0. Before coding | Secrets, data location, and the decisions |
| 1. Build phases | What to build, in which order, and when each phase is done |
| 2. Tests per node | For each of the 11 nodes: what it receives and what the test checks |
| 3. Graph tests | Routing, follow-ups, no loops |
| 4. Tool tests | The deterministic functions, one by one |
| 5. End-to-end tests | The 4 example questions on real data |
| 6. Claude evaluation | Accuracy of the Claude calls |
| 7. Tooling | Libraries and commands |
| 8. Definition of done | The checklist for handing in |

---

## 0. Before coding

### 0.1 Secrets (where the OAG key goes)
1. Create `Wonderful- FDE/.env`. Never commit it, and never paste keys in chat:
   ```
   OAG_API_KEY=...
   ANTHROPIC_API_KEY=...
   ```
2. Commit only `.env.example`, with empty values. `.gitignore` includes `.env`, `data/raw/` and `data/cache/`.
3. Code reads keys with `pydantic-settings`. A missing key fails at start-up with a clear message; keys are never logged or printed (tested).

### 0.2 Data location
The BTS files are in `03_Projects/`, one level up. Move them to `Wonderful- FDE/data/raw/` so the project is self-contained.

### 0.3 Decisions
All decisions are made. Details are in ARCHITECTURE.md.

| # | Decision | Section |
|---|---|---|
| D1 | **Passenger counts (T-100) come from the data.bts.gov API** (Socrata dataset `r495-tyji`, no key; checked: identical to the provided file). The on-time data and the FAA forecast have no API we found, so they are official downloads; the T-100 file is the fallback if the API is down | §5.1 |
| D2 | **OAG use (decided after the Phase 0 spike):** the key is the free tier (10 records per call; no seats or actual times). OAG is used **only for long-haul share** of the airport a question names; the 4 constraint signals come from BTS public data | §5.1, [docs/oag_notes.md](docs/oag_notes.md) |
| D3 | Long haul: block time (scheduled gate-to-gate time) ≥ 6 h by default; the ≥ 3,000 mi number is always shown too | §6.6 |
| D4 | **Claude Haiku 5.5** for all Claude calls, through the Anthropic API (`CLAUDE_MODEL`, `LLM_PROVIDER=anthropic`); effort set per call; switchable in config (Opus 5.5, or OpenRouter) | §7 |
| D5 | No "turned away today" (spill) term | §6.2 design note |
| D6 | The 4 signals in `C` are scaled by percentile rank | §6.3 |
| D7 | Growth comes from the FAA Terminal Area Forecast | §6.2 |
| D8 | 11 nodes; planning, fetching and checking happen in one `get_data` node | §4 |
| D9 | Airports with no FAA forecast use a fallback growth of 2% a year (configurable) and carry a caveat, instead of being left unscored | §6.2, §6.9 |
| D10 | **Gateway:** the Anthropic API (default now). OpenRouter stays available as an option (`LLM_PROVIDER=openrouter`, `OPENROUTER_MODEL`) | §7.4 |
| D11 | **Web search has three roles:** external facts for research questions; background for "why" questions; gaps the tools cannot fill. Every web fact carries its source and date and is reported as a web source, not the agent's data. Never used in scores or rankings | §4.2, §5.1 |
| D12 | **Model choice:** `CLAUDE_MODEL=claude-haiku-5-5` (configured). Price from the Anthropic pricing page, checked 2026-11-05: $0.10 input / $0.50 output per million tokens for prompts up to 100,000 tokens; higher above that. Evidence: scope evaluation 17/17 before the answer-mode rewrite. Changing the model means re-running `tests/evals/run_scope_eval.py` and the live e2e | §7.0 |

---

## 1. Build phases

### Phase 0 — OAG spike (≈ 45 min)
Goal: replace assumptions about OAG with facts **before** writing the client.
- [ ] `scripts/oag_spike.py`: one call for one airport's departures on one day.
- [ ] Record: base URL, auth header, required parameters, pagination, rate limit and quota, the date range allowed (past and future), and the fields available: block time, distance, aircraft, **seats?**, codeshare markers, **flight status with actual times?**
- [ ] Save 2–3 real responses (keys removed) to `tests/fixtures/oag/`. All OAG client tests use them.
- [ ] Write the findings to `docs/oag_notes.md`. If a field is missing, update the sources in ARCHITECTURE §5.1 and §6.3.
- [ ] Set the per-question OAG call budget from the subscription's quota.
- **Done when:** we know which signals OAG can feed, and how many requests one airport-week costs.

### Phase 1 — Data foundation (≈ 2 h)
- [ ] `build_bts_aggregates.py` (pandas): remove exact duplicate flights; on-time → airport × date × hour cells (departures, arrivals, taxi-out and taxi-in sums and counts) and airport × month figures; T-100 → clean numbers ("2,700" → 2700), dates, `AIRPORT_ID` kept. Output: Parquet in `data/processed/`.
- [ ] `build_taf.py`: download the latest FAA TAF database (XLSX, faa.gov/data_research/aviation/taf) → enplanements per airport and fiscal year, actual and forecast → `data/processed/taf.parquet`, with the edition recorded.
- [ ] `build_taf_error.py`: download older TAF editions and compare their forecasts with actual enplanements → forecast error by hub size and horizon, used for the growth range.
- [ ] `build_airport_reference.py`: BTS id ↔ code ↔ name ↔ city ↔ state ↔ location ↔ FAA hub size, from the FAA airport list in the TAF download.
- [ ] `config/regions.yaml` and `config/aliases.yaml`.
- [ ] `calibrate_signal_scales.py`: compute the 4 signals for every US airport that passes the hard checks, and store each signal's sorted reference values with the date → `data/reference/signal_reference.parquet`.
- [ ] Use the `analysis/` scripts as the reference implementation: production code must reproduce `analysis/output/airport_signals_v3.csv` (test).
- [ ] Backtest (if time allows): compute RPS with the TAF edition of 2018 and check that the top airports did grow over 2018–2019.

### Phase 2 — Deterministic tools (≈ 2.5 h)
In this order; each step only uses earlier ones.
1. [x] `resolve.py`: airport lookup and region lists.
2. [x] `checks.py`: the hard checks (ARCHITECTURE §6.4).
3. [x] `oag_client.py`: cache, pagination, retries per OAG's documentation, budget, codeshare filter, key sent to the OAG host only.
4. [x] `public_store.py`: reads the processed BTS and TAF Parquet files.
5. [x] `metrics.py`: one function per metric and signal, including `busy_minus_quiet`.
6. [x] `scoring.py`: `growth_share`, `ΔD`, percentile-ranked signals, `C`, RPS, ranking, measured range, caveats, hub size, user parameters.
7. [x] `facts.py`: the facts bundle for each request type.
8. [x] `output_guard.py`: number grounding check and template answer.
9. [x] `get_data` node: public-data coverage checks; long haul from OAG with the BTS fallback.
- **Done when:** `pytest tests/unit` passes, and `python -m airport_agent.cli rank --region "New England"` prints a ranked table with no Claude call.

### Phase 3 — LangGraph wiring (≈ 2 h)
- [ ] `state.py` (TypedDict + Pydantic models), `builder.py` (the 11 nodes and their edges), `routing.py` (the routing rules of ARCHITECTURE §4.2).
- [ ] Code nodes = thin wrappers around the tools.
- [ ] Claude nodes with the `anthropic` SDK, structured outputs, prompts in `llm/prompts/`.
- [ ] SQLite checkpointer; one `thread_id` per chat session.
- **Done when:** `tests/nodes` and `tests/graph` pass with Claude mocked, and the 4 example questions run end-to-end with the real Claude.

### Phase 4 — Chat interface (≈ 1 h)
- [ ] Streamlit chat + side panel (score table, metrics with source / period / caveats, parameters in use, node trace).
- [ ] Voice (only if time allows): `st.audio_input` → `faster-whisper` → the same graph.

### Phase 5 — Evaluation and documents (≈ 1 h)
- [ ] Run the end-to-end tests for the 4 questions and their follow-ups; save the transcripts.
- [ ] Short design document (2–3 pages) from ARCHITECTURE.md: scoring methodology, trade-offs, where AI is used.
- [ ] README: setup, `.env`, data build, run, test.

**Cut list if time runs out** (cut the first item first):
1. Voice.
2. The 2018 backtest.
3. The measured growth range from past TAF editions. Fallback: show only the `C` range, and say the growth range is missing.
4. User adjustments in follow-ups (ARCHITECTURE §6.8).
5. Web context for "why" questions.

---

## 2. Tests per node (`tests/nodes/`)

**Where examples belong:** concrete cases live **here and in `tests/`**, never in ARCHITECTURE.md, so the code follows the rules and not the examples. A **held-out set** in `tests/heldout/` (other airports, regions and phrasings that appear in no document or prompt) must pass too. A lint test fails the build if any airport code or city name is hard-coded in `src/`.

**Every node test has the same shape:** given an input state → run the node alone → check (a) the keys it wrote, (b) that it wrote no other keys, (c) the values. A shared helper `assert_only_writes(node, state, allowed_keys)` checks (a) and (b).

| # | Node | Input state | Writes | The test checks |
|---|---|---|---|---|
| 1 | `understand_request` (Claude mocked) | Each example question; follow-ups; "how do you score?"; an off-topic question; an IRR request; a mixed in-scope + out-of-scope request; injection attempts | request, counters | Request type and metrics are right; `methodology` and `out_of_scope` detected; a mixed request keeps the in-scope type and fills `out_of_scope_parts`; schema failure → 1 retry → "please rephrase" |
| 2 | `resolve_airports` | "LA and Santa Ana", "New England", "Anchorage", "SFO", "Portland", "Xyzzy" | airport_refs | Right status and candidates for each; no invented codes; every New England airport present, small ones included |
| 3 | `disambiguate_airports` (Claude mocked) | "LA" next to "Santa Ana"; "Portland" with no context; "Xyzzy" not found | airport_refs, clarification | LAX chosen; Portland → `ask_user` with exactly PDX and PWM in the question; not found → `ask_user`; a non-candidate answer is rejected |
| 4 | `get_data` (HTTP mocked) | New England airports; a long-haul question with OAG ok / OAG failing / OAG over budget | raw_data, data_quality, counters | Complete T-100 → `ok`, otherwise `missing` with the reason; OAG failure or over-budget day → BTS fallback with the reason; cached day → no calls |
| 5 | `compute_metrics` | Small hand-built datasets | metrics | Exact values; null in → null out with the reason; sum over sum; means weighted by flights |
| 6 | `score_and_rank` | Metric fixtures for 3 airports; a parameter change | scores | Hand-computed RPS matches; `C = 0` → RPS 0; `ΔD` shown but never used in RPS; hub size matches the FAA bands; no airport removed for size or caveats; range low ≤ base ≤ high; the parameter change re-scores without new data |
| 7 | `prepare_results` | A score table; a methodology request; a same-results follow-up | facts | Snapshot test; assumptions present; methodology facts equal the config values; the follow-up uses `context` only |
| 8 | `web_context` (mocked) | A "why" request; a non-"why" request; a research request; a search result containing instructions; a failed search | web_context, web_status | Runs only for "why", research or data gaps; every note keeps its source and date; a failed search is reported as failed; injected instructions ignored; nothing reaches `metrics` |
| 9 | `explain` (Claude mocked, plus live eval) | facts | draft_answer | Numbers only from facts (checked by `output_guard`); web context in its own labeled part |
| 10 | `output_guard` | A good draft; an invented number; an extra airport; a missing section; two failures in a row | answer, or route back to `explain` | Correct accept / reject; the second failure gives the template answer, which is never empty and contains every number in `facts` |
| 11 | `respond` | An answer; a clarification; a scope reply | messages, context | The next turn can use `context`; after a clarification, the user's reply resolves the airport |

---

## 3. Graph tests (`tests/graph/`)
- **Routing:** every row of the routing table (ARCHITECTURE §4.2) has a test:
  - out of scope → scope reply; unreadable twice → "please rephrase"
  - methodology → `prepare_results`
  - same airports, metrics, period and parameters as `context` → `prepare_results`, with no fetch
  - only parameters changed → `score_and_rank`, with no fetch
  - any difference, or empty `context` → the normal path
  - unresolved airport → `disambiguate_airports` → chosen, or ask the user
  - "why" → `web_context`
  - grounding fails twice → template answer
- **No loops:** a source that always fails is marked down, and the turn still ends with a partial answer; the only cycle is one `explain` retry.
- **Reuse is safe:** a follow-up that differs from `context` in any airport, metric or period never reuses stored results; the mocked APIs record zero calls on reuse turns.
- **Follow-ups** (several turns with the checkpointer): "Which New England airports…?" → "Why is the first one first?" (no fetch) → "Use a 10-year horizon" (re-score only) → "Compare it with Hartford" ("it" = the first airport).
- **Consistency:** an airport gets the same numbers in rank, compare and explain answers.

---

## 4. Tool tests (`tests/unit/`)
- **resolve:** exact code (any case), exact name, name variants ("San Francisco Int'l."), alias, a misspelling, a weak fuzzy match counts as ambiguous, ALL-CAPS words aren't mistaken for codes, metro names are ambiguous, at most 3 candidates, stable order.
- **regions:** every region lists all its airports, small ones included, each with its FAA hub size.
- **period:** the default period is the latest 12 months covered by both BTS datasets; its fiscal year starts the growth horizon.
- **oag_client:** auth header and default parameters; pagination until there is no `next`; a relative `next` is resolved against the API host; the key is never sent to another host and never appears in errors; 429 → wait and retry as documented; cache reuse; codeshare copies point to the operating flight; missing fields are null, not zero.
- **public_store:** year-over-year needs complete matching months; an airport whose code changed is still joined by ID.
- **checks:** each hard check passes and fails at its boundary (29 / 30 flights; load factor 1.0 / 1.01; 11 / 12 months).
- **long-haul fallback:** an OAG error or a day over budget gives the BTS (domestic) long haul, with the reason as a caveat.
- **metrics:** long-haul share is per departure, not from an average distance; an "all flights" long-haul question is refused on domestic-only data; `busy_minus_quiet` matches a hand-computed value on synthetic hourly cells; no busy or no quiet hours → null with the reason; peak saturation uses the airport's own 99th-percentile hour; cancellation causes add up.
- **growth (TAF):** `growth_share = TAF(base + H) / TAF(base) − 1` matches a hand-computed value; base year = the fiscal year the data window ends in; a 10-year horizon reads the 10-year-ahead value; an airport missing from TAF → fallback growth, `growth_share = 1.02^H − 1` (2% a year, from config), with the "fallback growth" caveat; changing the fallback in config changes only those airports; load factor missing → not scored, with the reason.
- **signal scaling:** percentile rank matches a hand-computed value, ties counting half; the reference median gives 0.5; above every reference value gives 1, below every one gives 0; the same airport gets the same value in different questions; values for the 105 test airports reproduce `analysis/output/airport_signals_v3.csv`.
- **scoring:** `C` = mean of the 4 scaled signals (weights rescaled to sum to 1); RPS rises when growth or `C` rises; growth ≤ 0 or C = 0 → RPS 0, no rank, listed with its reason ("FAA forecast shows no growth…" or "constraint C is 0…"); a missing signal → `C` from the rest + caveat; a horizon other than 5 / 10 / 20 is rejected; negative weights are rejected.
- **ranking:** RPS = 100 × max(0, growth_share) × C matches a hand-computed value; sorted by RPS; nothing capped; the worked example ranks small very tight > big tight > biggest roomy; airports of every hub size are ranked; airports that can't be scored are listed with the reason.
- **range:** the growth range from past TAF editions matches a hand-computed value on a small fixture; the `C` bootstrap with a fixed seed is reproducible.
- **hub size:** an airport at exactly 1.00%, 0.25% and 0.05% of US enplanements falls in the higher class, as 49 U.S.C. 47102 defines it.
- **output_guard:** number extraction handles "12.3%", "1,234", "2.1M" and "+5 pts"; rounding tolerance; airport code check.
- **assumptions register:** every number in `scoring.yaml` appears in ARCHITECTURE §6.9 and the other way round, so no constant can be added silently.

---

## 5. End-to-end tests (`tests/golden/`, real data, recorded OAG responses)

| Question | The answer must contain |
|---|---|
| New England terminal expansion | RPS ranking; forecast extra passengers; growth and `C` drivers; measured range; FAA hub size; caveats; every New England airport that passes the checks (none dropped for size); assumptions |
| LA vs Santa Ana congestion | LA resolved to LAX, with that choice stated; the 4 constraint signals and `C` side by side |
| Anchorage long-haul % | The definition used; both definitions' numbers; a note on domestic vs international coverage |
| SFO unmet demand and why | FAA forecast extra demand `ΔD`, `C` and its drivers (the same numbers as the RPS inputs); no "passengers lost" count; a note that today's turned-away demand can't be measured; FAA growth labeled as planned, not unmet; web context kept separate with its sources |

---

## 5b. End-to-end tests in a browser (`tests/e2e/`, Playwright)
The real Streamlit app, started on a free port with its own temporary cache, driven in Chromium.
- **Offline (default, free):** a scripted stand-in for Claude (`AIRPORT_AGENT_CLAUDE=fake`, `src/airport_agent/llm/fake.py`) that reads requests with patterns, never with airport names.
  - the home page shows the four example questions
  - a ranking from an example button: the answer, the side-panel trace (all data nodes ran), the score table, the horizon
  - follow-ups: "why is the first one first?" runs no data nodes; "10-year horizon" only re-scores
  - a city with two airports asks the user with exactly its candidates; a metro name is resolved from context
  - long haul shows both definitions and 0 OAG calls (cached); a "why" question adds web context with its sources
  - out-of-scope and methodology questions skip the data nodes
  - the chat box is disabled while an answer is written; "New conversation" resets the chat
  - every test fails on any browser console error
- **Live (`RUN_LIVE_E2E=1`, about a minute):** the same flow with the real Claude; the draft must pass the number check without the template.

## 5c. Module changes (this round: scope, flags, last-resort lookup, model switch)

Each row says what changed, why, and the test that covers it. "Config" = a file in `config/`.

| Module | Change | Why | Covered by |
|---|---|---|---|
| `settings.py` | `LLM_PROVIDER` (anthropic / openrouter), `CLAUDE_MODEL`, `OPENROUTER_MODEL`, `OPENROUTER_API_KEY`; cache folder override for tests | Model and gateway are configuration, not code | test_openrouter (provider switch) |
| `llm/client.py` | Anthropic client: `web_search_tool(model)` (basic tool `web_search_20250305` for Haiku; `web_search_20260209` for Opus/Sonnet, per the docs checked 2026-11-05); `web_lookup` (gap lookup); plain-language API errors | Haiku is not listed for dynamic filtering in the docs, so it gets the basic tool; a missing key or no credit must not blame the question | test_scope (API errors), e2e |
| `llm/openrouter.py` | New option: the same five calls through OpenRouter; strict schema from Pydantic; key never logged | Alternative gateway, not the default | test_openrouter (12 tests) |
| `llm/schemas.py` | Intent: rank, compare, metric_lookup, explain_why, conceptual, research, methodology, out_of_scope (`airport_info` removed); `rank_by`, `order`, `top_n` (required, null = not stated) | Answer modes; rank by any measure; "top N" | test_answer_modes, test_scope, scope eval |
| `llm/fake.py` | Scripted Claude for offline tests: new intents, `web_lookup` | Offline e2e and graph tests | e2e (offline tier) |
| `llm/prompts/understand.md` | Rewritten around the domain, the answer modes and financial rules; no example questions; non-US places are out of scope | General scope, not memorised examples | scope eval (run before this rewrite: 17/17; not re-run) |
| `llm/prompts/explain.md`, `disambiguate.md`, `web_context.md`, `web_lookup.md`, `_provenance.md` (new) | Explain by answer mode; web status rules; shared provenance rules (primary sources, labels for reported fact / source claim / interpretation, dates, disagreements reported, never guessed) | Spec: prompts must be general; retrieved text is untrusted data | test_answer_modes (no-claim and invented-citation checks) |
| `llm/prompts/explain.md` | Ranked-by and scope statements; flagged airports; the web-lookup section | Honest presentation of rankings and web facts | number-check tests |
| `llm/prompts/web_lookup.md` (new) | Web search: official sources, cited, no guessing | For facts the tools lack and research questions | test_graph_flows (lookup flows) |
| `graph/nodes.py` | `normalize_request` (code rules on Claude's parse); `score_and_rank` scores without ranges, then ranges for shown airports only; `prepare_results` by intent; `web_context` node sets `web_status` (results / no_reliable_result / failed) and `web_error`; `_audit`; OpenRouter/Anthropic factory | Scope rules in code; speed; honest web status | test_graph_flows, test_answer_modes, test_scope |
| `graph/routing.py` | Conceptual and methodology → `prepare_results` (no tools); research → resolve only if a place is named; reuse of earlier results only for rank, compare, explain_why, metric_lookup with the same airports and region; a new external fact is never reused | Answer modes; no stale reuse | test_answer_modes |
| `graph/builder.py` | Unchanged topology (11 nodes) | | |
| `tools/resolve.py` | `is_national`, `_region_key` (ignore generic words), national scope, `is_metro_alias` | "US", "Pacific division", metro-vs-state questions | test_scope (region names), held-out |
| `tools/scoring.py` | `rank(by, order)` over any measure; `rank_value`; `measure()` | Rank by size, congestion, growth, signals, delays | test_scope (rank by measure) |
| `tools/facts.py` | `ranked_by`, `scope`, `signals_measured` ("2 of 4 ⚑"), `flag_note`, `info_facts`, `data_gaps` | Flags for small airports; general questions | test_graph_flows (flags) |
| `tools/output_guard.py` | Number and airport checks by answer mode; web-status claims must match the search that ran; template for rankings, flags, the labeled web section | Grounding per answer type; no false claims about searches | test_answer_modes, test_metrics_scoring (template) |
| `tools/facts.py` | `answer_mode` on every facts bundle (computed, conceptual, research, methodology); `score_formula_not_used` in computed facts | Answer-mode validation | test_answer_modes |
| `tools/scoring.py` | `drivers(...)` returns C, percentiles, signals, growth and caveats for an airport; no RPS | "Why" answers without a score | test_answer_modes (why does not compute RPS) |
| `ui/app.py` | Chat box disabled while answering; "For: <question>" in the side panel; signals column; errors not shown as "rephrase"; example questions from config | Browser-tested behaviour | e2e (12 tests, stable over 8 runs) |
| `scripts/build_bts_aggregates.py` | T-100 from the data.bts.gov API (same dataset as the file; verified identical), file as fallback | No manual download needed | build check |
| `config/regions.yaml` | `national` names (incl. "US states"), `ignore_words` | Scope wording | test_scope |
| `config/answer_text.yaml` | Example questions; assumptions text | Moved out of code (place names) | test_rules (no place names in code) |
| `config/kpi_catalog.yaml` | `research: []` (web only, no computed metric) | Research route | test_answer_modes |
| `config/answer_text.yaml` | Scope reply lists what is covered and what is not; methodology text states the corrections (screening index, not profitability; FAA growth is not unmet demand; taxi-in is indirect; taxi-out does not prove the fix; peak throughput is not certified capacity; 30-flight rule of thumb; same-data validation is not independent) | Methodology corrections | test_answer_modes (number rules) |
| `config/output_guard.yaml` | Acronyms that are not airport codes | Grounding check | number-check tests |
| `tests/evals/run_scope_eval.py` (new) | Live scope evaluation: 17 questions, plus 10 unseen phrasings with `--unseen` | Measures understanding, not only code | Haiku 5.5, before the answer-mode rewrite: 17/17; unseen set 25/27 (2 failures: "rank London airports" and "quietest airports in the whole country"). Not re-run since the rewrite |
| `tests/unit/test_scope.py`, `test_openrouter.py` (new) | Scope rules, ranking by measure, API errors; OpenRouter client | | part of the 131 offline tests |
| `tests/graph/test_answer_modes.py` (new) | 19 tests: conceptual answers use no tools; "why" does not compute RPS; a current airport fact → research; published budget → research; IRR projection → out_of_scope reply; supported compare without score; new external fact not reused; explain reuses results; failed search reported as failed; research keeps source link; number-check rules per mode; wrong-airport and invented-citation rejected; claimed search rejected | Answer-mode spec | 19 tests, mocked Claude |

### 5d. Answer-mode round (2026-11-05): status

**Implemented (offline):** the answer modes and their routing; web results with source, date and status; output guard by mode; the five prompts rewritten without example questions, with shared provenance rules; facts and drivers by mode; methodology corrections in config and text; the scope reply listing what is covered and not; UI web-source panel showing "did not run" and "no reliable source" honestly.

**Tests run (this round, actual results):** `pytest --ignore=tests/e2e`: 131 passed. `pytest tests/e2e`: 12 passed, 1 skipped (live, needs `RUN_LIVE_E2E=1`). Ruff: the safe auto-fixes were applied to `src` and `tests`; 20 findings remain in `src`/`tests` (style rules such as `dict()` literals and one `hub != hub` check in `graph/nodes.py`), and 45 in `analysis/` and `scripts/`, which were not changed.

**Not verified live (no live run in this round):**
- The rewritten prompts and the new answer modes against the live Haiku 5.5 model. The last live scope evaluation was before the rewrite.
- The web search tool on Haiku 5.5 (`web_search_20250305`). The docs confirm Haiku 5.5 supports web search and that dynamic filtering is documented for other models; the code path has not been called live.
- OpenRouter web plugin behaviour (mocked only).

### 5e. Evidence-first, round 2: status

**Implemented, tested offline:** initial computed answers keep the "Assumptions & uncertainty" section and the screening sentence; a focused follow-up (`facts.follow_up`) does not repeat them, but must state each essential caveat (`facts.essential_caveats`). The output check and the fallback template use the same rules. The explain prompt forbids suggesting a data error from equal rounded values. A turn that only discusses airports keeps the earlier ranking population and display count (`respond`). Zero-RPS airports are listed, not ranked. Ranked airports outside the shown count are listed in `ranked_not_shown`.

**Tests (actual results):** 157 offline tests pass, including the new conversation test (rank → why → 10-year horizon) and the case where the "why" turn is not flagged as a follow-up. E2E: 12 passed, 1 skipped (live).

**One live run (real Haiku 5.5, authorized budget, run once):** turn 1 ranked 13 airports and showed 10, with no guard problems. Turn 2 ("why is BOS above BTV") ran no data nodes and no web search; `top_n` stayed empty; one draft was rejected by the guard and rewritten (the reason was not recorded). Turn 3 ("10-year horizon") re-scored without fetching data and kept 13 in scope and 10 shown. Answer lengths: 157 words for turn 2 and 136 for turn 3, above the 60–120 word target. This is a single run, so model behaviour is only partly verified.

### 5f. Wrong-airport research: status

**What was checked first.** The running server on port 8501 had started before the last several code changes (its Python modules were loaded once and not reloaded). Prompts are read from disk on every call, so the server mixed new prompts with old code. The server was restarted so it runs the current code.

**Root cause.** A research question about "the top-ranked airport" had no link to the stored ranking. The parser sometimes left the airports empty, so the web query had no target, and sometimes left the phrase as an airport mention, which fell through to disambiguation, where the model could choose an unrelated airport. Fixes: a `ranked_positions` field resolved in code from the stored ranking; generic phrases dropped from the mentions; a clarification instead of a search when there is no ranking; the research fallback names its airport and carries no investment disclaimer.

**Also fixed.** Comparative and superlative claims in computed answers are checked against the airport rows (a claim that BOS had "the highest growth" when another airport had more was not caught before). The research check accepts figures from the retrieved text. A score explanation runs no data step and no web search (checked in the running app).

**Tests.** 178 offline tests pass, including the wrong-airport regression (it fails without the resolver), the generic-phrase case, the position and no-ranking cases, and the comparison checks. E2E: 12 passed, 1 skipped (live).

**Live check (running app, restarted, three questions).** "Rank New England airports": ranked, BOS first. "What expansion projects are currently planned at the top-ranked airport?": routed resolve → prepare → web → explain, with no disambiguation; the search was for BOS; the answer is titled for Boston Logan. "Explain the score: why is the first airport ranked above the second?": no data step and no web search; the explanation attributes the order to forecast growth and C; 133 words, one guard retry (reason not recorded).

**Not verified.** The model's wording beyond these checks: the research answer still labels its items "unverified", which the provenance rules discourage. The length target (60–120 words) is met only roughly.

**Planned, not built:** the growth range from past TAF editions (`build_taf_error.py`); a backtest; voice; the live evaluation set of about 40 questions (§6).

---

## 6. Claude evaluation (`tests/evals/`, run by hand, costs money)
- About 40 questions: paraphrases of the 4 examples, follow-ups, ambiguous names, out-of-scope questions, injection attempts.
- Graded on: request type and airport accuracy (exact match), grounding pass rate (`output_guard`), and a rubric for clarity and stated assumptions.
- Target before the demo: request type ≥ 95%, airports 100%, grounding 100%.

---

## 7. Tooling
`uv` · `pytest` + `pytest-mock` + `respx` (HTTP mocks) · `ruff` · `pandas` + `pyarrow` (Parquet) · `pydantic` / `pydantic-settings` · `rapidfuzz` · `langgraph` + `langgraph-checkpoint-sqlite` · `anthropic` · `streamlit`.

Commands: `make data` (aggregates + TAF + reference + signal scales) · `make test` · `make run`.

---

## 8. Definition of done
- [ ] The 4 example questions are answered correctly end-to-end, with follow-ups.
- [ ] Every number in an answer traces back to a tool output (`output_guard` on).
- [ ] The scoring is documented and reproducible from config; every chosen number is in ARCHITECTURE §6.9.
- [x] Unit, node, graph and held-out tests pass; scope evaluation saved (`tests/evals/scope_eval_results.json`).
- [ ] Design document (scoring methodology, trade-offs, AI usage) and README are done.
