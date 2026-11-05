"""Held-out cases: airports, regions and phrasings that appear in no document or prompt.

They check that the code follows the rules, not the examples it was written around.
"""

from pathlib import Path

import pytest

from airport_agent.tools import public_store as ps
from airport_agent.tools.resolve import expand_region, resolve_one
from airport_agent.tools.scoring import rank, score_airport

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT / "data/reference/signal_reference.parquet").exists(),
                                reason="processed data not built")


@pytest.mark.parametrize("text,code", [("Denver", "DEN"), ("DTW", "DTW"), ("Seattle Tacoma International", "SEA"),
                                       ("Nashville", "BNA"), ("Salt Lake City", "SLC")])
def test_heldout_airports_resolve(text, code):
    r = resolve_one(text)
    assert r.status == "resolved" and r.airport.iata == code


def test_heldout_metro_is_ambiguous():
    r = resolve_one("Chicago")
    assert r.status == "ambiguous" and {"ORD", "MDW"} <= {c.iata for c in r.candidates}


@pytest.mark.parametrize("region,must_include", [("Mountain", {"DEN", "PHX", "SLC"}), ("Texas", {"DFW", "IAH", "AUS"}),
                                                 ("Midwest", {"ORD", "MSP", "DTW"})])
def test_heldout_regions_expand(region, must_include):
    assert must_include <= {c.iata for c in expand_region(region)}


def test_heldout_region_ranks_with_no_airport_dropped():
    airports = expand_region("Mountain")
    scores = rank([score_airport(a.airport_id, with_range=False) for a in airports])
    assert len(scores) == len(airports)
    ranked = [s for s in scores if s.rank]
    assert ranked and all(a.rps >= b.rps for a, b in zip(ranked, ranked[1:], strict=False))
    assert all(s.rank is None for s in scores if not s.scored or not s.rps)


def test_heldout_scores_are_stable_across_questions():
    den = int(ps.airports().set_index("iata").loc["DEN", "airport_id"])
    assert score_airport(den, with_range=False).rps == score_airport(den, with_range=False).rps
