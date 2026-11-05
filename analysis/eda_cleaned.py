"""EDA part 3: clean delay rate and taxi-out the same way as taxi-in, then re-compare.

Usage:
    python analysis/eda_cleaned.py <data_dir> [out_dir]

Needs output/airport_signals_v2.csv from eda_gate_side.py (same out_dir).

Cleaned = value in busy departure hours (util >= 0.75) minus value in quiet hours
(util <= 0.25) at the same airport; util = departures in the hour / the airport's own
p99 hourly departures. Removes what does not depend on how busy the airport is
(airfield size for taxi-out; airline and weather problems for delays).

Check for time of day: busy hours are often late in the day, when delays have built up
through the day. A within-hour version compares, for the same hour of day, busier days
with quieter days (regression with hour-of-day fixed effects).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = pd.read_csv(os.path.join(OUT, "airport_signals_v2.csv"), index_col=0)
BUSY, QUIET = 0.75, 0.25

use = ["ORIGIN_AIRPORT_ID", "FL_DATE", "DEP_TIME_BLK", "TAXI_OUT", "DEP_DEL15", "CANCELLED",
       "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "DEST_AIRPORT_ID"]
files = sorted(glob.glob(os.path.join(DATA, "T_ONTIME_REPORTING_*.csv")))
f = pd.concat([pd.read_csv(x, usecols=use) for x in files], ignore_index=True).drop_duplicates()
f = f[f.ORIGIN_AIRPORT_ID.isin(df.index)]

cell = (f.groupby(["ORIGIN_AIRPORT_ID", "FL_DATE", "DEP_TIME_BLK"])
        .agg(deps=("CANCELLED", "size"), taxi=("TAXI_OUT", "mean"), n_taxi=("TAXI_OUT", "count"),
             del15=("DEP_DEL15", "mean"), n_del=("DEP_DEL15", "count"))
        .reset_index())
cell["util"] = (cell.deps / cell.ORIGIN_AIRPORT_ID.map(cell.groupby("ORIGIN_AIRPORT_ID").deps.quantile(0.99))).clip(upper=1.5)


def busy_minus_quiet(c, y, n):
    c = c[c[n] > 0]
    b, q = c[c.util >= BUSY], c[c.util <= QUIET]
    if b[n].sum() == 0 or q[n].sum() == 0:
        return np.nan
    return np.average(b[y], weights=b[n]) - np.average(q[y], weights=q[n])


def within_hour_effect(c, y, n):
    """Slope of y on util with hour-of-day fixed effects, x (BUSY - QUIET) -> same units as busy-minus-quiet."""
    c = c[c[n] > 0].copy()
    w = c[n].astype(float)
    for col in (y, "util"):
        m = (c[col] * w).groupby(c.DEP_TIME_BLK).transform("sum") / w.groupby(c.DEP_TIME_BLK).transform("sum")
        c[col + "_d"] = c[col] - m
    den = (w * c.util_d ** 2).sum()
    return np.nan if den == 0 else (w * c.util_d * c[y + "_d"]).sum() / den * (BUSY - QUIET)


rows = {}
for ap, c in cell.groupby("ORIGIN_AIRPORT_ID"):
    rows[ap] = {
        "taxi_out_excess": busy_minus_quiet(c, "taxi", "n_taxi"),
        "delay_excess": busy_minus_quiet(c, "del15", "n_del"),
        "taxi_out_excess_fe": within_hour_effect(c, "taxi", "n_taxi"),
        "delay_excess_fe": within_hour_effect(c, "del15", "n_del"),
    }
df = df.join(pd.DataFrame.from_dict(rows, orient="index"))
df.to_csv(os.path.join(OUT, "airport_signals_v3.csv"))
pd.set_option("display.width", 200, "display.max_columns", 30)

print("=== A. Cleaned signals: distribution ===")
NEWC = ["taxi_out_excess", "delay_excess", "taxi_out_excess_fe", "delay_excess_fe"]
print(df[NEWC].describe(percentiles=[.1, .5, .9]).round(3).T, "\n")

print("=== B. Raw vs cleaned, and simple vs within-hour version (Spearman) ===")
for a, b in [("taxi_out_median", "taxi_out_excess"), ("delay15_rate", "delay_excess"),
             ("taxi_out_excess", "taxi_out_excess_fe"), ("delay_excess", "delay_excess_fe")]:
    print(f"{a:18s} vs {b:20s} rho={df[a].corr(df[b], method='spearman'):+.2f}")
print()

print("=== C. Examples ===")
show = ["code", "taxi_out_median", "taxi_out_excess", "delay15_rate", "delay_excess", "delay_excess_fe", "taxi_in_excess"]
print(df.set_index("code").loc[["ORD", "LGA", "SAN", "SFO", "BOS", "PHX", "MCO", "HPN", "BZN", "TUL", "MSP", "LAS"], show[1:]].round(3), "\n")

OUTCOMES = ["runway_slope", "gate_slope", "combined"]
print("=== D. Each signal vs congestion (Spearman; * = built from the same data as that outcome) ===")
CIRC = {"runway_slope": {"taxi_out_median", "taxi_out_excess", "taxi_out_excess_fe"},
        "gate_slope": {"taxi_in_median", "taxi_in_excess"}}
SIG = ["taxi_out_median", "taxi_out_excess", "taxi_out_excess_fe", "delay15_rate", "delay_excess",
       "delay_excess_fe", "taxi_in_median", "taxi_in_excess", "load_factor", "peak_saturation"]
out = []
for s in SIG:
    r = {"signal": s}
    for o in OUTCOMES:
        circ = s in CIRC.get(o, set()) or (o == "combined" and any(s in v for v in CIRC.values()))
        r[o] = f"{df[s].corr(df[o], method='spearman'):+.2f}{'*' if circ else ''}"
    out.append(r)
print(pd.DataFrame(out).to_string(index=False), "\n")


def norm01(s):
    lo, hi = s.quantile(0.1), s.quantile(0.9)
    return ((s - lo) / (hi - lo)).clip(0, 1)


BASE = ["load_factor", "peak_saturation"]
SCHEMES = {
    "A plan today": {"load_factor": .25, "peak_saturation": .25, "nas_share": .20, "taxi_out_median": .15, "gauge_vs_flights": .15},
    "C current pick (raw delay, raw taxi-out)": dict.fromkeys(BASE + ["delay15_rate", "taxi_out_median", "taxi_in_excess"], .2),
    "E cleaned delay, raw taxi-out": dict.fromkeys(BASE + ["delay_excess", "taxi_out_median", "taxi_in_excess"], .2),
    "G raw delay, cleaned taxi-out": dict.fromkeys(BASE + ["delay15_rate", "taxi_out_excess", "taxi_in_excess"], .2),
    "F all cleaned": dict.fromkeys(BASE + ["delay_excess", "taxi_out_excess", "taxi_in_excess"], .2),
}
N = pd.DataFrame({s: norm01(df[s]) for s in {k for v in SCHEMES.values() for k in v}})
C = pd.DataFrame({k: sum(N[s] * w for s, w in v.items()) for k, v in SCHEMES.items()})

rng = np.random.default_rng(0)
idx = [rng.integers(0, len(df), len(df)) for _ in range(1000)]


def rho(x, y, i=None):
    if i is not None:
        x, y = x[i], y[i]
    return pd.Series(x).corr(pd.Series(y), method="spearman")


print("=== E. Each C vs congestion (Spearman, bootstrap 90% interval) ===")
rows = []
for k in SCHEMES:
    r = {"scheme": k}
    for o in OUTCOMES:
        b = [rho(C[k].values, df[o].values, i) for i in idx]
        r[o] = f"{rho(C[k].values, df[o].values):.2f} [{np.percentile(b, 5):.2f}, {np.percentile(b, 95):.2f}]"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False), "\n")

print("=== F. Difference vs 'C current pick' (bootstrap) ===")
ref = "C current pick (raw delay, raw taxi-out)"
for k in ["E cleaned delay, raw taxi-out", "G raw delay, cleaned taxi-out", "F all cleaned"]:
    for o in OUTCOMES:
        d = np.array([rho(C[k].values, df[o].values, i) - rho(C[ref].values, df[o].values, i) for i in idx])
        print(f"{k:32s} {o:13s} diff={d.mean():+.2f} [{np.percentile(d, 5):+.2f}, {np.percentile(d, 95):+.2f}] better in {np.mean(d > 0):.0%}")
print()
print("=== G. Rank agreement between schemes ===")
print(C.corr(method="spearman").round(2))
