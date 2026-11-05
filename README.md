# Airport Investment Intelligence Agent

A chat agent for analysts who screen US airports for terminal renovation. It ranks airports by
**FAA forecast passenger growth × how capacity-constrained the airport is today** (the Renovation Profitability
Score, a screening index), compares airports, and answers conceptual, methodology and sourced web questions.

Computed numbers come from deterministic code. Claude reads the question, resolves ambiguous airport names,
writes the answer, and adds web research where a question needs external facts. Before an answer is shown, its
numbers and claims are checked against the facts for its answer mode.

## Documents

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: the design, the rendered agent graph (§4.1), routing, state, scoring method and
  assumptions register.
- **[DESIGN.md](DESIGN.md)**: a shorter design summary, with the same rendered graph (§3).
- **[PLAN.md](PLAN.md)**: the build plan, the tests, and the status of every phase. Section 5d/5e/5f record what was
  verified live and what was not.
- How the scoring was chosen: [analysis/FINDINGS_constraint_weights.md](analysis/FINDINGS_constraint_weights.md)
- OAG API findings: [docs/oag_notes.md](docs/oag_notes.md)

## Status

| Part | Status |
|---|---|
| Data build, metrics, RPS scoring, ranking | Implemented; tested offline with unit and held-out tests |
| LangGraph agent, routing, follow-up reuse, output guard, fallback | Implemented; tested offline with a scripted model |
| Chat UI, per-answer execution details | Implemented; tested in the browser with a scripted model |
| Conceptual and methodology answers with the real model | Verified live (one conceptual question through the UI, about 8 s) |
| Rankings, research and follow-ups with the real model | Verified live on an earlier version; the newest answer modes are not yet re-verified live |
| Growth range from past FAA forecast errors | Planned, not built (the range shown covers the constraint only) |
| Voice input | Planned, not built |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install langgraph langgraph-checkpoint-sqlite anthropic pydantic pydantic-settings rapidfuzz pandas pyarrow pyyaml httpx respx pytest streamlit openpyxl pytest-playwright
.venv/bin/playwright install chromium     # only for the browser tests
cp .env.example .env                      # then set ANTHROPIC_API_KEY (OAG_API_KEY is optional)
```

The default model call goes to the Anthropic API with `claude-haiku-5-5` (`LLM_PROVIDER=anthropic`). OpenRouter is
an option (`LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY`). Never commit `.env`.

## Data

Put the BTS on-time files in `data/raw/` (`T_ONTIME_REPORTING_*.csv`), and the unpacked FAA TAF download
(`APO100_TAF_Final_2025.zip` from taf.faa.gov) in `data/raw/taf_2025/`. Passenger data (T-100) is pulled from
the data.bts.gov API by the build (no key needed); `data/raw/AFF-T100_*.csv` is only a fallback. Then:

```bash
.venv/bin/python scripts/build_bts_aggregates.py
.venv/bin/python scripts/build_taf.py
.venv/bin/python scripts/build_airport_reference.py
.venv/bin/python scripts/calibrate_signal_scales.py
```

## Run

```bash
.venv/bin/streamlit run src/airport_agent/ui/app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501). The input is at the top, the newest
question is shown first, and each answer has a "How this answer was built" panel with its execution details.

Ranking without Claude (command line):

```bash
PYTHONPATH=src .venv/bin/python -m airport_agent.cli rank --region "New England"
```

## Test

```bash
.venv/bin/python -m pytest tests -q
```

This runs the unit, graph, held-out and design-rule tests, and the end-to-end browser tests (Playwright). The
graph and browser tests use a scripted stand-in for Claude (`AIRPORT_AGENT_CLAUDE=fake`), so they run offline, without
an API key, and use their own temporary cache. The live test is skipped unless you enable it:

```bash
RUN_LIVE_E2E=1 .venv/bin/python -m pytest tests/e2e -m live -q
```

The live test calls the real model and costs a few cents.
