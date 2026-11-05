# Airport Investment Intelligence Agent

A chat agent that helps analysts find US airports where a terminal renovation would be most profitable:
**FAA forecast passenger growth × how capacity-constrained the airport is today**. Every number comes from
code; Claude only reads the question, resolves ambiguous airport names, gathers labeled web context for
"why" questions, and writes the answer — which is then checked number by number.

- Design: [ARCHITECTURE.md](ARCHITECTURE.md) · Build plan and tests: [PLAN.md](PLAN.md)
- How the scoring was chosen: [analysis/FINDINGS_constraint_weights.md](analysis/FINDINGS_constraint_weights.md)
- OAG API findings: [docs/oag_notes.md](docs/oag_notes.md)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install langgraph langgraph-checkpoint-sqlite anthropic pydantic pydantic-settings rapidfuzz pandas pyarrow pyyaml httpx respx pytest streamlit openpyxl pytest-playwright
.venv/bin/playwright install chromium     # browser for the end-to-end tests
cp .env.example .env        # then fill in OAG_API_KEY and OPENROUTER_API_KEY
```

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

Ranking without Claude (command line):

```bash
PYTHONPATH=src .venv/bin/python -m airport_agent.cli rank --region "New England"
```

## Test

```bash
.venv/bin/python -m pytest tests -q
```

This runs everything except the live test: unit tests, graph tests, held-out cases, design-rule tests,
and **end-to-end tests** that start the real Streamlit app and drive it in a real browser (Playwright).
The graph and end-to-end tests use a scripted stand-in for Claude (`AIRPORT_AGENT_CLAUDE=fake`), so they
run offline, for free, without an API key. They use their own temporary cache, so they never touch your
conversation memory or audit log.

Live end-to-end test with the real Claude (about a minute, a few cents):

```bash
RUN_LIVE_E2E=1 .venv/bin/python -m pytest tests/e2e -m live -q
```
