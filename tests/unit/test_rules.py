"""Design rules that must hold for the whole code base (PLAN.md §2, §4)."""

import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "airport_agent"
BASES = ("Official", "Published", "Tested", "Judgment", "Definition", "Computation", "From the OAG plan")


@pytest.mark.skipif(not (ROOT / "data/reference/airports.parquet").exists(), reason="reference not built")
def test_no_airport_codes_or_city_names_hard_coded_in_src():
    """All place-specific knowledge lives in config/, never in code or prompts."""
    ref = pd.read_parquet(ROOT / "data/reference/airports.parquet")
    hub = ref[ref.hub_size.isin(["large", "medium", "small"])]
    codes = set(hub.iata)
    acronyms = set(yaml.safe_load((ROOT / "config/output_guard.yaml").read_text())["acronyms_that_are_not_airports"])
    cities = {c for c in hub.city.dropna() if len(c) > 5}
    offenders = []
    for path in list(SRC.rglob("*.py")) + list(SRC.rglob("*.md")):
        text = path.read_text()
        for code in re.findall(r"\b[A-Z]{3}\b", text):
            if code in codes and code not in acronyms:
                offenders.append(f"{path.relative_to(ROOT)}: {code}")
        offenders += [f"{path.relative_to(ROOT)}: {c}" for c in cities if re.search(rf"\b{re.escape(c)}\b", text)]
    assert not offenders, "place-specific names in code: " + ", ".join(sorted(set(offenders)))


def test_every_scoring_number_states_its_basis():
    """Each number in scoring.yaml has a basis comment on its line or on its section line (ARCHITECTURE §6.9)."""
    missing, section_basis = [], False
    for line in (ROOT / "config/scoring.yaml").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        has_basis = any(b in line for b in BASES)
        if not line.startswith(" "):               # top-level section
            section_basis = has_basis
            continue
        value = line.split("#")[0].split(":", 1)[1].strip() if ":" in line else ""
        if re.search(r"\d", value) and not (has_basis or section_basis or line.startswith("    ")):
            missing.append(line.strip())
    assert not missing, "numbers without a stated basis: " + "; ".join(missing)


def test_fallback_growth_and_horizons_match_the_docs():
    cfg = yaml.safe_load((ROOT / "config/scoring.yaml").read_text())
    arch = (ROOT / "ARCHITECTURE.md").read_text()
    assert f"{cfg['growth']['fallback_growth_per_year']:.0%} a year" in arch
    assert cfg["horizon"]["allowed_years"] == [5, 10, 20] and "5, 10 or 20 years" in arch
