"""Part 3b: is the cleaned delay rate confounded by time of day, and is delay rate needed at all?
Usage: python analysis/eda_cleaned_check.py [out_dir]   (needs airport_signals_v3.csv)"""
import os, sys
import numpy as np, pandas as pd
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = pd.read_csv(os.path.join(OUT, "airport_signals_v3.csv"), index_col=0)
print("Mean delay_excess (busy minus quiet hours):", round(df.delay_excess.mean(), 3),
      "-> negative = quiet hours have MORE delays (late-evening build-up), i.e. time of day dominates")
print("delay15_rate vs taxi_out_excess rho:", round(df.delay15_rate.corr(df.taxi_out_excess, method="spearman"), 2))
def norm01(s):
    lo, hi = s.quantile(.1), s.quantile(.9); return ((s - lo) / (hi - lo)).clip(0, 1)
B = ["load_factor", "peak_saturation", "taxi_out_excess", "taxi_in_excess"]
S = {"G raw delay, cleaned taxis": B + ["delay15_rate"],
     "H within-hour delay, cleaned taxis": B + ["delay_excess_fe"],
     "I no delay rate (4 signals)": B}
N = pd.DataFrame({s: norm01(df[s]) for s in set(sum(S.values(), []))})
C = pd.DataFrame({k: N[v].mean(axis=1) for k, v in S.items()})
rng = np.random.default_rng(0); idx = [rng.integers(0, len(df), len(df)) for _ in range(1000)]
r = lambda x, y, i: pd.Series(x[i]).corr(pd.Series(y[i]), method="spearman")
for k in S:
    line = f"{k:36s}"
    for o in ["runway_slope", "gate_slope", "combined"]:
        b = [r(C[k].values, df[o].values, i) for i in idx]
        line += f"  {o}={C[k].corr(df[o], method='spearman'):.2f} [{np.percentile(b,5):.2f},{np.percentile(b,95):.2f}]"
    print(line)
for k in list(S)[1:]:
    d = np.array([r(C[k].values, df.combined.values, i) - r(C["G raw delay, cleaned taxis"].values, df.combined.values, i) for i in idx])
    print(f"{k} vs G on combined: diff={d.mean():+.2f} [{np.percentile(d,5):+.2f},{np.percentile(d,95):+.2f}]")
print("rank agreement:\n", C.corr(method="spearman").round(2))
