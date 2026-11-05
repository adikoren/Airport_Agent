"""Part 6: are the chosen cut-offs safe? Sensitivity of C to:
  - the busy / quiet hour cut-offs (base 0.75 / 0.25)
  - the percentile used as an airport's hourly "maximum" (base 99th)

Usage:
    python analysis/eda_sensitivity.py <data_dir> [out_dir]
Needs output/airport_signals_v3.csv (for the airport list, load factor and the
congestion measures used as the fixed yardstick).

For each variant, recompute peak saturation, cleaned taxi-out and cleaned taxi-in,
build C (percentile rank, equal weights) and compare with the base:
  - Spearman between the variant's C ranking and the base C ranking
  - how many airports move more than 5 places
  - C vs the combined congestion measure (kept fixed at the base definition)
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
base = pd.read_csv(os.path.join(OUT, "airport_signals_v3.csv"), index_col=0)
ids = set(base.index)

use = ["ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "DEP_TIME_BLK", "CRS_ARR_TIME", "TAXI_OUT", "TAXI_IN",
       "CANCELLED", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM"]
files = sorted(glob.glob(os.path.join(DATA, "T_ONTIME_REPORTING_*.csv")))
f = pd.concat([pd.read_csv(x, usecols=use) for x in files], ignore_index=True).drop_duplicates()

dep = (f[f.ORIGIN_AIRPORT_ID.isin(ids)]
       .groupby(["ORIGIN_AIRPORT_ID", "FL_DATE", "DEP_TIME_BLK"])
       .agg(n=("CANCELLED", "size"), y=("TAXI_OUT", "mean"), w=("TAXI_OUT", "count")).reset_index()
       .rename(columns={"ORIGIN_AIRPORT_ID": "ap"}))
a = f[f.DEST_AIRPORT_ID.isin(ids)].copy()
a["hr"] = (a.CRS_ARR_TIME // 100) % 24
arr = (a.groupby(["DEST_AIRPORT_ID", "FL_DATE", "hr"])
       .agg(n=("CANCELLED", "size"), y=("TAXI_IN", "mean"), w=("TAXI_IN", "count")).reset_index()
       .rename(columns={"DEST_AIRPORT_ID": "ap"}))
print(f"departure cells: {len(dep):,}  arrival cells: {len(arr):,}")


def signals(ceiling_q, busy, quiet):
    out = {}
    for name, c in [("dep", dep), ("arr", arr)]:
        cap = c.groupby("ap").n.quantile(ceiling_q)
        u = c.n / c.ap.map(cap)
        if name == "dep":
            peak = c.groupby(["ap", "FL_DATE"]).n.max().groupby(level=0).mean() / cap
            out["peak_saturation"] = peak
        cc = c[c.w > 0].assign(u=u[c.w > 0])
        b, q = cc[cc.u >= busy], cc[cc.u <= quiet]
        bm = (b.y * b.w).groupby(b.ap).sum() / b.w.groupby(b.ap).sum()
        qm = (q.y * q.w).groupby(q.ap).sum() / q.w.groupby(q.ap).sum()
        out["taxi_out_excess" if name == "dep" else "taxi_in_excess"] = bm - qm
    s = pd.DataFrame(out)
    s["load_factor"] = base.load_factor
    return s


S = ["load_factor", "peak_saturation", "taxi_out_excess", "taxi_in_excess"]


def C_of(s):
    return s[S].rank(pct=True).mean(axis=1)


variants = [("base: 99th pct, 75/25", .99, .75, .25),
            ("busy/quiet 70/30", .99, .70, .30), ("busy/quiet 80/20", .99, .80, .20),
            ("busy/quiet 85/15", .99, .85, .15), ("busy/quiet 65/35", .99, .65, .35),
            ("ceiling 95th pct", .95, .75, .25), ("ceiling 97th pct", .97, .75, .25),
            ("ceiling max (100th)", 1.0, .75, .25)]
res, Cs = [], {}
for name, cq, b, q in variants:
    s = signals(cq, b, q)
    C = C_of(s).reindex(base.index)
    Cs[name] = C
    res.append((name, int(C.isna().sum()), C.corr(base.combined, method="spearman")))
ref = Cs[variants[0][0]]
rows = []
for name, missing, rho_cong in res:
    C = Cs[name]
    ok = C.notna() & ref.notna()
    rk_ref, rk = ref[ok].rank(ascending=False), C[ok].rank(ascending=False)
    rows.append({"variant": name, "airports without a value": missing,
                 "rank agreement with base": round(C[ok].corr(ref[ok], method="spearman"), 3),
                 "airports moving > 5 places": int(((rk - rk_ref).abs() > 5).sum()),
                 "largest move (places)": int((rk - rk_ref).abs().max()),
                 "top-10 overlap with base": len(set(rk[rk <= 10].index) & set(rk_ref[rk_ref <= 10].index)),
                 "C vs congestion": round(rho_cong, 2)})
pd.set_option("display.width", 220, "display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))
