"""Unit tests for metrics, checks and scoring (PLAN.md §4). Synthetic data, hand-computed answers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airport_agent.tools import checks
from airport_agent.tools import metrics as m
from airport_agent.tools import scoring as sc
from airport_agent.tools.metrics import MetricValue
from airport_agent.tools.public_store import Period

P = Period(pd.Timestamp("2025-08-01"), pd.Timestamp("2026-07-01"))
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- checks

def test_enough_flights_boundary():
    assert checks.enough_flights(29, "x") is not None
    assert checks.enough_flights(30, "x") is None


def test_load_factor_possible_boundary():
    assert checks.load_factor_possible(1.0) is None
    assert checks.load_factor_possible(1.01) is not None


def test_complete_months():
    assert checks.complete_months(11) is not None
    assert checks.complete_months(12) is None


# ---------------------------------------------------------------- T-100 metrics

def _t100(months=12, pax=80, seats=100):
    idx = pd.date_range("2025-08-01", periods=months, freq="MS")
    return pd.DataFrame({"airport_id": 1, "month": idx, "passengers": pax, "seats": seats,
                         "departures": 1, "intl_passengers": 20})


def test_load_factor_is_sum_over_sum():
    t = _t100()
    t.loc[0, ["passengers", "seats"]] = [10, 1000]       # one big, empty month
    lf = m.load_factor(t, 1, P)
    assert lf.value == pytest.approx((10 + 11 * 80) / (1000 + 11 * 100))


def test_load_factor_missing_month_is_missing_not_zero():
    lf = m.load_factor(_t100(months=11), 1, P)
    assert lf.value is None and "11 of 12" in lf.missing_reason


def test_impossible_load_factor_is_missing():
    assert m.load_factor(_t100(pax=120, seats=100), 1, P).value is None


# ---------------------------------------------------------------- hourly signals

def _cells(spec):
    """spec: list of (date, hour, departures, taxi_out_sum, taxi_out_n)."""
    rows = [dict(airport_id=1, date=pd.Timestamp(d), hour=h, departures=dep, taxi_out_sum=ts, taxi_out_n=tn,
                 arrivals=dep, taxi_in_sum=ts, taxi_in_n=tn) for d, h, dep, ts, tn in spec]
    return pd.DataFrame(rows)


def _busy_quiet_cells(busy_taxi=20.0, quiet_taxi=10.0, days=40):
    spec = []
    for i in range(days):
        d = pd.Timestamp("2025-09-01") + pd.Timedelta(days=i)
        spec.append((d, 8, 10, busy_taxi * 10, 10))   # 10 departures = the maximum hour → busy
        spec.append((d, 22, 1, quiet_taxi * 1, 1))    # 1 departure = 10% of maximum → quiet
    return _cells(spec)


def test_busy_minus_quiet_hand_computed():
    v = m.taxi_out_excess(_busy_quiet_cells(), 1, P)
    assert v.value == pytest.approx(10.0)
    assert v.detail["busy_flights"] == 400 and v.detail["quiet_flights"] == 40


def test_busy_minus_quiet_can_be_negative():
    assert m.taxi_out_excess(_busy_quiet_cells(busy_taxi=8, quiet_taxi=10), 1, P).value == pytest.approx(-2.0)


def test_busy_minus_quiet_needs_30_flights_per_group():
    v = m.taxi_out_excess(_busy_quiet_cells(days=29), 1, P)   # 29 quiet flights
    assert v.value is None and "29 flights" in v.missing_reason


def test_peak_saturation_hand_computed():
    v = m.peak_saturation(_busy_quiet_cells(), 1, P)
    assert v.value == pytest.approx(1.0)                      # every day's peak = the maximum hour


def test_percentile_rank_ties_count_half():
    ref = np.array([1, 2, 2, 3])
    assert m.percentile_rank(2, ref) == pytest.approx((1 + 0.5 * 2) / 4)
    assert m.percentile_rank(0, ref) == 0.0
    assert m.percentile_rank(9, ref) == 1.0


def test_long_haul_counts_flights_not_average_distance():
    flights = [dict(block_minutes=400, miles=3500, passenger=True, international=True)] + \
              [dict(block_minutes=60, miles=300, passenger=True, international=False)] * 3
    v = m.long_haul_from_flights(flights, "day")
    assert v.value == pytest.approx(0.25)


def test_long_haul_missing_block_time_is_excluded_and_stated():
    flights = [dict(block_minutes=None, miles=4000, passenger=False, international=True),
               dict(block_minutes=400, miles=3500, passenger=True, international=True)]
    v = m.long_haul_from_flights(flights, "day")
    assert v.value == 1.0 and v.detail["block_time_missing"] == 1 and v.caveats


# ---------------------------------------------------------------- scoring

def _ref():
    rows = [{"signal": s, "value": float(v)} for s in sc.SIGNALS for v in range(10)]
    return pd.DataFrame(rows)


def _sig(values):
    return {s: MetricValue(values.get(s), "x", "test", "p", missing_reason=None if s in values else "missing")
            for s in sc.SIGNALS}


def test_C_is_mean_of_percentiles():
    con = sc.constraint(_sig({s: 5.0 for s in sc.SIGNALS}), ref=_ref())
    assert pytest.approx(0.55) == con.C          # 5 of 10 below + half of 1 tie = 0.55


def test_missing_signal_uses_the_rest_and_adds_caveat():
    con = sc.constraint(_sig({"load_factor": 9.0, "peak_saturation": 9.0}), ref=_ref())
    assert pytest.approx(0.95) == con.C and len(con.caveats) == 2


def test_weights_rescaled_and_validated():
    w = sc.weights({"load_factor": 1, "peak_saturation": 1, "taxi_out_excess": 1, "taxi_in_excess": 1})
    assert sum(w.values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        sc.weights({"load_factor": -1})


def _taf(rows):
    return pd.DataFrame([dict(locid="XXX", fiscal_year=y, enplanements=e, edition="TAF test") for y, e in rows])


def test_growth_from_taf_reads_the_horizon_year():
    taf = _taf([(2026, 100.0), (2031, 115.0), (2036, 130.0)])
    assert sc.growth("XXX", 5, P, taf).growth_share == pytest.approx(0.15)
    assert sc.growth("XXX", 10, P, taf).growth_share == pytest.approx(0.30)   # no constant-rate assumption


def test_growth_fallback_when_faa_has_no_forecast():
    g = sc.growth("NONE", 5, P, _taf([(2026, 100.0)]))
    assert g.source == "fallback" and g.growth_share == pytest.approx(1.02 ** 5 - 1) and g.caveats


def test_fiscal_year_of_window():
    assert P.fiscal_year == 2026
    assert Period(pd.Timestamp("2025-10-01"), pd.Timestamp("2026-10-01")).fiscal_year == 2027


def test_worked_example_ranking_order():
    """ARCHITECTURE §6.2 worked example: small very tight > big tight > biggest roomy."""
    rps = {"small": 100 * 0.338 * 0.9, "big": 100 * 0.159 * 0.7, "biggest": 100 * 0.217 * 0.1}
    assert sorted(rps, key=rps.get, reverse=True) == ["small", "big", "biggest"]


def test_rank_puts_unscored_and_no_growth_last():
    def s(i, rps, scored=True):
        return sc.AirportScore(i, f"A{i}", "n", None, None, scored, None, rps, None, None, 0.1, "x", None, None,
                               None, None, 0.5, None, {}, None, [])
    out = sc.rank([s(1, 2.0), s(2, None, scored=False), s(3, 5.0), s(4, 0.0)])
    assert [a.airport_id for a in out][:2] == [3, 1] and out[0].rank == 1 and out[2].rank is None


# ---------------------------------------------------------------- reproduces the analysis

@pytest.mark.skipif(not (ROOT / "data/processed/hourly_cells.parquet").exists(), reason="processed data not built")
def test_signals_reproduce_the_analysis():
    from airport_agent.tools import public_store as ps
    v3 = pd.read_csv(ROOT / "analysis/output/airport_signals_v3.csv", index_col=0)
    sample = v3.sample(25, random_state=0)
    cells, t = ps.hourly_cells(), ps.t100_monthly()
    for ap, row in sample.iterrows():
        assert m.load_factor(t, ap, ps.default_period()).value == pytest.approx(row.load_factor, abs=1e-9)
        assert m.taxi_out_excess(cells, ap, ps.default_period()).value == pytest.approx(row.taxi_out_excess, abs=1e-9)
        assert m.taxi_in_excess(cells, ap, ps.default_period()).value == pytest.approx(row.taxi_in_excess, abs=1e-9)
        # peak saturation: the first analysis script de-duplicated on a column subset and merged a few
        # distinct flights, so allow a small difference (the production build de-duplicates on full rows)
        assert m.peak_saturation(cells, ap, ps.default_period()).value == pytest.approx(row.peak_saturation, abs=0.03)


def test_framing_numbers_are_in_the_facts():
    """'the last 12 months' and 'four signals' are true statements and must pass the number check."""
    from airport_agent.tools import facts as F
    from airport_agent.tools import output_guard as G
    f = F.common("Aug 2025 – Jul 2026", 5, [], None)
    draft = ("Over the last 12 months, using 4 signals over 5 years. ### Assumptions & uncertainty\n"
             + f["disclaimer"])
    assert G.check(draft, f) == []


def test_template_handles_every_question_type():
    """The fixed-format fallback answer must render a readable answer for each request type, never raw JSON."""
    from airport_agent.tools import facts as F
    from airport_agent.tools import output_guard as G
    common = F.common("Aug 2025 – Jul 2026", 5, [], None)
    for qt in ("rank", "explain_why"):
        f = {"question_type": qt, "region": None, "ranking": [], "listed_not_ranked": [], **common}
        out = G.template(f)
        assert "```" not in out and "Assumptions & uncertainty" in out
