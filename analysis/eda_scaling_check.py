"""Part 4: how should the 4 signals in C be put on one scale?
Compares the tested 10th-90th percentile scaling (caps at both ends) with plain min-max
(no caps) and percentile rank. Usage: python analysis/eda_scaling_check.py [out_dir]"""
import os, sys
import numpy as np, pandas as pd
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = pd.read_csv(os.path.join(OUT, "airport_signals_v3.csv"), index_col=0)
S = ["load_factor", "peak_saturation", "taxi_out_excess", "taxi_in_excess"]
def p10p90(s): lo, hi = s.quantile(.1), s.quantile(.9); return ((s - lo) / (hi - lo)).clip(0, 1)
def minmax(s): return (s - s.min()) / (s.max() - s.min())
def pct(s): return s.rank(pct=True)
for name, f in [("10-90 percentile, capped (tested)", p10p90), ("min-max, no caps", minmax), ("percentile rank", pct)]:
    N = pd.DataFrame({s: f(df[s]) for s in S}); C = N.mean(axis=1)
    capped = int(((N == 0) | (N == 1)).sum().sum())
    print(f"--- {name}: C vs combined congestion rho={C.corr(df.combined, method='spearman'):.2f}, "
          f"values stuck at exactly 0 or 1: {capped} of {N.size}")
    print("   median of each scaled signal:", N.median().round(2).to_dict())
mm = pd.DataFrame({s: minmax(df[s]) for s in S}); mm["code"] = df.code
print("\nMost extreme airport per signal (min-max):")
for s in S:
    top2 = df.nlargest(2, s)
    print(f"  {s:16s} max={df[s].max():.2f} ({top2.code.iloc[0]}), next={top2[s].iloc[1]:.2f} ({top2.code.iloc[1]}), median={df[s].median():.2f}")

# bootstrap interval for the chosen method (percentile rank) and its gain over the old capped one
rng = np.random.default_rng(0)
Cp = pd.DataFrame({s: pct(df[s]) for s in S}).mean(axis=1).values
Co = pd.DataFrame({s: p10p90(df[s]) for s in S}).mean(axis=1).values
y = df.combined.values
r = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
bp, bd = [], []
for _ in range(1000):
    i = rng.integers(0, len(df), len(df))
    bp.append(r(Cp[i], y[i])); bd.append(r(Cp[i], y[i]) - r(Co[i], y[i]))
print(f"\npercentile rank: rho={r(Cp, y):.2f}, 90% interval [{np.percentile(bp,5):.2f}, {np.percentile(bp,95):.2f}]")
print(f"gain over old capped scaling: {np.mean(bd):+.2f}, 90% interval [{np.percentile(bd,5):+.2f}, {np.percentile(bd,95):+.2f}], better in {np.mean(np.array(bd)>0):.0%}")
