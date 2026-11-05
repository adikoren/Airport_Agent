"""Phase 1: reference values for the percentile rank of the 4 constraint signals.

Usage:
    python scripts/calibrate_signal_scales.py

Computes the 4 signals for every US airport that passes the hard checks, over the default
period, and writes data/reference/signal_reference.parquet (airport_id, iata, signal, value,
period, calibrated_at). Percentile rank in scoring is computed against these values.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airport_agent.settings import get_settings  # noqa: E402
from airport_agent.tools import public_store as ps  # noqa: E402
from airport_agent.tools.scoring import SIGNALS, compute_signals  # noqa: E402


def main() -> None:
    period = ps.default_period()
    ref = ps.airports().set_index("airport_id")
    ids = sorted(ps.hourly_cells().airport_id.unique())
    rows, skipped = [], {s: 0 for s in SIGNALS}
    for ap in ids:
        sig = compute_signals(int(ap), period)
        for s in SIGNALS:
            if sig[s].value is None:
                skipped[s] += 1
                continue
            rows.append({"airport_id": int(ap), "iata": ref.loc[ap, "iata"], "signal": s, "value": sig[s].value,
                         "period": period.label, "calibrated_at": date.today().isoformat()})
    out = pd.DataFrame(rows)
    path = get_settings().reference_dir / "signal_reference.parquet"
    out.to_parquet(path, index=False)
    print(f"period {period.label}; {len(ids)} airports with hourly data")
    print("reference airports per signal:", out.groupby("signal").airport_id.nunique().to_dict())
    print("airports left out (failed a hard check):", skipped)


if __name__ == "__main__":
    main()
