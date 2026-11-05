"""Command line: rank airports without any Claude call (Phase 2 check).

    .venv/bin/python -m airport_agent.cli rank --region "New England" [--horizon 10]
"""

from __future__ import annotations

import argparse

from airport_agent.tools import public_store as ps
from airport_agent.tools.resolve import expand_region
from airport_agent.tools.scoring import rank, score_airport


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank")
    r.add_argument("--region", required=True)
    r.add_argument("--horizon", type=int, default=None)
    a = p.parse_args()
    airports = expand_region(a.region)
    if airports is None:
        raise SystemExit(f'Unknown region "{a.region}". Use a US state or a Census region/division.')
    scores = rank([score_airport(x.airport_id, horizon=a.horizon) for x in airports])
    print(f"{a.region}: {ps.default_period().label}, horizon {a.horizon or 5} years\n")
    print(f"{'rank':>4}  {'airport':5} {'hub size':8} {'RPS':>6} {'range':>11} {'growth':>7} {'C':>5}")
    for s in scores:
        rng = f"{s.rps_low:.1f}–{s.rps_high:.1f}" if s.rps_low is not None else "-"
        rps = f"{s.rps:.1f}" if s.rps is not None else "-"
        print(f"{s.rank or '-':>4}  {s.iata:5} {s.hub_size or '-':8} {rps:>6} {rng:>11} "
              f"{s.growth_share:7.1%} {s.C if s.C is None else round(s.C, 2)!s:>5}"
              + ("" if s.rank else f"   ({s.not_scored_reason or 'no demand growth forecast'})"))


if __name__ == "__main__":
    main()
