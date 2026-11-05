"""EDA part 2: re-check the constraint signals against runway AND gate congestion.

Usage:
    python analysis/eda_gate_side.py <data_dir> [out_dir]

Needs output/airport_signals.csv from eda_constraint_weights.py (same out_dir).

Adds:
- gate_slope: per airport, weighted regression of hourly mean taxi-in on arrival utilisation
  (scheduled arrivals in the hour / the airport's own p99 hourly arrivals). Steep = gates fill up.
- taxi_in_excess: mean taxi-in in busy arrival hours (util >= 0.75) minus quiet hours
  (util <= 0.25) at the same airport. Removes the "big airfield = long taxi" effect.
- combined congestion = mean of the runway and gate slope ranks (one index, C is not split).
- Spearman agreement of C under each weighting with each outcome, with bootstrap intervals.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = pd.read_csv(os.path.join(OUT, "airport_signals.csv"), index_col=0)


def wslope(y, x, w):
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)


use = ["DEST_AIRPORT_ID", "FL_DATE", "CRS_ARR_TIME", "TAXI_IN", "CANCELLED", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID"]
files = sorted(glob.glob(os.path.join(DATA, "T_ONTIME_REPORTING_*.csv")))
f = pd.concat([pd.read_csv(x, usecols=use) for x in files], ignore_index=True).drop_duplicates()
f = f[f.DEST_AIRPORT_ID.isin(df.index)]
f["arr_hour"] = (f.CRS_ARR_TIME // 100) % 24
print(f"Arrivals at the {len(df)} airports: {len(f):,}")

cell = (f.groupby(["DEST_AIRPORT_ID", "FL_DATE", "arr_hour"])
        .agg(arrs=("CANCELLED", "size"), taxi_in=("TAXI_IN", "mean"), n=("TAXI_IN", "count"))
        .reset_index())
cell = cell[cell.n > 0]
p99 = cell.groupby("DEST_AIRPORT_ID").arrs.quantile(0.99)
cell["util"] = (cell.arrs / cell.DEST_AIRPORT_ID.map(p99)).clip(upper=1.5)

gate_slope, excess = {}, {}
for ap, c in cell.groupby("DEST_AIRPORT_ID"):
    gate_slope[ap] = wslope(c.taxi_in.values, c.util.values, c.n.values)
    busy, quiet = c[c.util >= 0.75], c[c.util <= 0.25]
    if len(busy) and len(quiet):
        excess[ap] = np.average(busy.taxi_in, weights=busy.n) - np.average(quiet.taxi_in, weights=quiet.n)
df["gate_slope"] = pd.Series(gate_slope)
df["taxi_in_excess"] = pd.Series(excess)
df = df.dropna(subset=["gate_slope", "taxi_in_excess"])
df["runway_slope"] = df.congestion_slope
df["combined"] = (df.runway_slope.rank() + df.gate_slope.rank()) / 2
df.to_csv(os.path.join(OUT, "airport_signals_v2.csv"))
print(f"Airports with both measures: {len(df)}\n")
pd.set_option("display.width", 200, "display.max_columns", 30)

OUTCOMES = ["runway_slope", "gate_slope", "combined"]
print("=== A. The two congestion measures ===")
print(df[["runway_slope", "gate_slope", "taxi_in_excess"]].describe(percentiles=[.1, .5, .9]).round(2).T)
print("runway vs gate slope, Spearman:", round(df.runway_slope.corr(df.gate_slope, method="spearman"), 2))
print("\nTop 12 by gate slope:")
print(df.sort_values("gate_slope", ascending=False)[["code", "passengers", "gate_slope", "taxi_in_excess", "taxi_in_median", "runway_slope"]].head(12).round(2), "\n")

SIGNALS = ["load_factor", "peak_saturation", "delay15_rate", "taxi_out_median", "taxi_in_median",
           "taxi_in_excess", "nas_share", "gauge_vs_flights", "hours_at_capacity", "nas_min_per_dep",
           "months_lf_over_85", "passengers_vs_seats_growth", "nas_cancel_rate", "taxi_out_p90"]
CIRC = {"runway_slope": {"taxi_out_median", "taxi_out_p90"}, "gate_slope": {"taxi_in_median", "taxi_in_excess"}}
print("=== B. Each signal vs each congestion measure (Spearman rho; * = built from the same data) ===")
rows = []
for s in SIGNALS:
    r = {"signal": s}
    for o in OUTCOMES:
        rho = df[s].corr(df[o], method="spearman")
        mark = "*" if s in CIRC.get(o, set()) or (o == "combined" and any(s in v for v in CIRC.values())) else ""
        r[o] = f"{rho:+.2f}{mark}"
    rows.append(r)
print(pd.DataFrame(rows).sort_values("combined", ascending=False, key=lambda c: c.str[:5].astype(float)).to_string(index=False), "\n")


def norm01(s):
    lo, hi = s.quantile(0.1), s.quantile(0.9)
    return ((s - lo) / (hi - lo)).clip(0, 1)


N = pd.DataFrame({s: norm01(df[s]) for s in SIGNALS})
SCHEMES = {
    "A plan today": {"load_factor": .25, "peak_saturation": .25, "nas_share": .20, "taxi_out_median": .15, "gauge_vs_flights": .15},
    "B equal, raw taxi-in": {s: .2 for s in ["load_factor", "peak_saturation", "delay15_rate", "taxi_out_median", "taxi_in_median"]},
    "C equal, cleaned taxi-in": {s: .2 for s in ["load_factor", "peak_saturation", "delay15_rate", "taxi_out_median", "taxi_in_excess"]},
    "D regression weights (part 1)": {"peak_saturation": .306, "delay15_rate": .239, "load_factor": .199, "taxi_in_median": .056, "taxi_out_median": .2},
}
C = pd.DataFrame({k: sum(N[s] * w for s, w in v.items()) for k, v in SCHEMES.items()})

print("=== C. Agreement of each C ranking with each congestion measure (Spearman, bootstrap 90% interval) ===")
rng = np.random.default_rng(0)
idx = [rng.integers(0, len(df), len(df)) for _ in range(1000)]
rows = []
for k in SCHEMES:
    r = {"scheme": k}
    for o in OUTCOMES:
        point = C[k].corr(df[o], method="spearman")
        boot = [pd.Series(C[k].values[i]).corr(pd.Series(df[o].values[i]), method="spearman") for i in idx]
        r[o] = f"{point:.2f} [{np.percentile(boot, 5):.2f}, {np.percentile(boot, 95):.2f}]"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False), "\n")

print("=== D. Is each scheme better than 'A plan today' on the combined measure? (bootstrap) ===")
for k in list(SCHEMES)[1:]:
    diffs = []
    for i in idx:
        y = pd.Series(df.combined.values[i])
        diffs.append(pd.Series(C[k].values[i]).corr(y, method="spearman") - pd.Series(C["A plan today"].values[i]).corr(y, method="spearman"))
    diffs = np.array(diffs)
    print(f"{k:32s} diff={np.mean(diffs):+.2f}  90% interval [{np.percentile(diffs, 5):+.2f}, {np.percentile(diffs, 95):+.2f}]  share better={np.mean(diffs > 0):.2f}")
print()
print("=== E. Rank agreement between schemes ===")
print(C.corr(method="spearman").round(2))
