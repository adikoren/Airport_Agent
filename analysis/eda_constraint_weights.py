"""EDA: which capacity-constraint signals matter, and how should they be weighted?

Usage:
    python analysis/eda_constraint_weights.py <data_dir> [out_dir]

<data_dir> holds the BTS files (T_ONTIME_REPORTING_*.csv and the AFF-T100 summary).
Writes airport_signals.csv to out_dir (default analysis/output) and prints a report.

Idea: there is no label saying which airports are capacity-constrained, so we build one.
Queueing theory: near capacity, delay rises steeply with each extra flight. For every
airport we regress hourly mean taxi-out on hourly utilisation (departures in the hour /
the airport's own 99th-percentile hourly departures). The slope ("congestion
sensitivity") = extra taxi-out minutes going from an empty hour to a full hour.
Then we check which candidate signals predict that slope.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DATA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
MIN_PASSENGERS = 1_000_000
LAST = ("2025-08-01", "2026-07-01")
PRIOR = ("2024-08-01", "2025-07-01")


# ---------- small regression helpers (numpy) ----------
def ols(y, X, w=None):
    """Least squares with intercept. Returns coefs, HC1 robust p-values, R2."""
    from scipy import stats
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    sw = np.sqrt(np.ones(len(y)) if w is None else np.asarray(w, float))
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    n, k = Xw.shape
    r2 = 1 - (resid @ resid) / ((yw - np.average(y, weights=sw**2) * sw) ** 2).sum()
    bread = np.linalg.pinv(Xw.T @ Xw)
    meat = (Xw * resid[:, None] ** 2).T @ Xw
    cov = bread @ meat @ bread * n / max(n - k, 1)
    se = np.sqrt(np.diag(cov))
    p = 2 * (1 - stats.t.cdf(np.abs(beta / se), max(n - k, 1)))
    return beta, p, r2, n


# ---------- T-100 ----------
t100_file = glob.glob(os.path.join(DATA, "AFF-T100*.csv"))[0]
t = pd.read_csv(t100_file, skiprows=1, thousands=",")
t["d"] = pd.to_datetime(t["Date"], format="%Y %b %d %I:%M:%S %p")
t = t.rename(columns={"Total Passengers": "passengers", "Total Seats": "seats", "Total Departures": "deps",
                      "Outbound International Passengers": "intl_passengers"})


def window(df, w):
    m = df[(df.d >= w[0]) & (df.d <= w[1])]
    g = m.groupby("ORIGIN_AIRPORT_ID")[["passengers", "seats", "deps", "intl_passengers"]].sum(min_count=1)
    g["months"] = m.groupby("ORIGIN_AIRPORT_ID").size()
    return g


last, prior = window(t, LAST), window(t, PRIOR)
a = last.join(prior, rsuffix="_p")
a = a[(a.months == 12) & (a.months_p == 12) & (a.passengers >= MIN_PASSENGERS)].copy()
a["load_factor"] = a.passengers / a.seats
a["gauge_vs_flights"] = ((a.seats / a.deps) / (a.seats_p / a.deps_p) - 1) - (a.deps / a.deps_p - 1)
a["passengers_vs_seats_growth"] = (a.passengers / a.passengers_p - 1) - (a.seats / a.seats_p - 1)
lm = t[(t.d >= LAST[0]) & (t.d <= LAST[1])].copy()
lm["lf"] = lm.passengers / lm.seats
a["months_lf_over_85"] = (lm[lm.lf > 0.85].groupby("ORIGIN_AIRPORT_ID").size()
                          .reindex(a.index).fillna(0))
codes = t.drop_duplicates("ORIGIN_AIRPORT_ID", keep="last").set_index("ORIGIN_AIRPORT_ID")["Origin Airport Code"]
a["code"] = codes.reindex(a.index)
print(f"T-100: {len(a)} airports with >= {MIN_PASSENGERS:,} passengers and 24 complete months")

# ---------- On-time ----------
use = ["ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "DEP_TIME_BLK", "TAXI_OUT", "TAXI_IN",
       "CANCELLED", "CANCELLATION_CODE", "DEP_DEL15", "NAS_DELAY", "CARRIER_DELAY",
       "WEATHER_DELAY", "LATE_AIRCRAFT_DELAY"]
files = sorted(glob.glob(os.path.join(DATA, "T_ONTIME_REPORTING_*.csv")))
f = pd.concat([pd.read_csv(x, usecols=use) for x in files], ignore_index=True).drop_duplicates()
arr = f[f.DEST_AIRPORT_ID.isin(a.index)]
f = f[f.ORIGIN_AIRPORT_ID.isin(a.index)]
print(f"On-time: {len(f):,} departures at those airports, {len(files)} monthly files")

g = f.groupby("ORIGIN_AIRPORT_ID")
sig = pd.DataFrame({
    "flights_ontime": g.size(),
    "taxi_out_median": g.TAXI_OUT.median(),
    "taxi_out_p90": g.TAXI_OUT.quantile(0.9),
    "delay15_rate": g.DEP_DEL15.mean(),
    "nas_cancel_rate": g.CANCELLATION_CODE.apply(lambda s: (s == "C").sum()) / g.size(),
})
cause = g[["NAS_DELAY", "CARRIER_DELAY", "WEATHER_DELAY", "LATE_AIRCRAFT_DELAY"]].sum()
sig["nas_share"] = cause.NAS_DELAY / cause.sum(axis=1)
sig["nas_min_per_dep"] = cause.NAS_DELAY / sig.flights_ontime
sig["taxi_in_median"] = arr.groupby("DEST_AIRPORT_ID").TAXI_IN.median()

cell = (f.groupby(["ORIGIN_AIRPORT_ID", "FL_DATE", "DEP_TIME_BLK"])
        .agg(deps=("CANCELLED", "size"), taxi=("TAXI_OUT", "mean"), n_taxi=("TAXI_OUT", "count"))
        .reset_index())
p99 = cell.groupby("ORIGIN_AIRPORT_ID").deps.quantile(0.99).rename("p99_hourly")
cell = cell.join(p99, on="ORIGIN_AIRPORT_ID")
cell["util"] = cell.deps / cell.p99_hourly
daily_peak = cell.groupby(["ORIGIN_AIRPORT_ID", "FL_DATE"]).deps.max().groupby(level=0).mean()
sig["peak_saturation"] = daily_peak / p99
sig["hours_at_capacity"] = cell.assign(hot=cell.util >= 0.9).groupby("ORIGIN_AIRPORT_ID").hot.mean()

slopes = {}
for ap, c in cell[cell.n_taxi > 0].groupby("ORIGIN_AIRPORT_ID"):
    beta, p, r2, n = ols(c.taxi, c[["util"]].clip(upper=1.5), w=c.n_taxi)
    slopes[ap] = (beta[1], r2)
sig["congestion_slope"] = pd.Series({k: v[0] for k, v in slopes.items()})
sig["slope_r2"] = pd.Series({k: v[1] for k, v in slopes.items()})

df = a.join(sig, how="inner")
CURRENT = ["load_factor", "peak_saturation", "nas_share", "taxi_out_median", "gauge_vs_flights"]
NEW = ["taxi_in_median", "taxi_out_p90", "nas_min_per_dep", "delay15_rate", "nas_cancel_rate",
       "hours_at_capacity", "months_lf_over_85", "passengers_vs_seats_growth"]
SIGNALS = CURRENT + NEW
TAXI_OUT_BASED = ["taxi_out_median", "taxi_out_p90"]  # part of the outcome -> circular
df = df.dropna(subset=SIGNALS + ["congestion_slope"])
df.to_csv(os.path.join(OUT, "airport_signals.csv"))
print(f"Final table: {len(df)} airports\n")
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 60)

print("=== 1. Distribution of signals (p10 / p90 = candidate score anchors) ===")
print(df[SIGNALS + ["congestion_slope", "slope_r2"]].describe(percentiles=[.1, .5, .9]).round(3).T, "\n")

print("=== 2. Spearman correlation (redundancy check; |rho| > 0.7 = near-duplicate) ===")
corr = df[SIGNALS + ["congestion_slope"]].corr(method="spearman").round(2)
print(corr, "\n")
pairs = [(i, j, corr.loc[i, j]) for k, i in enumerate(SIGNALS) for j in SIGNALS[k + 1:] if abs(corr.loc[i, j]) > 0.7]
print("near-duplicate pairs:", pairs or "none", "\n")

z = (df[SIGNALS] - df[SIGNALS].mean()) / df[SIGNALS].std()
pca = PCA().fit(z)
print("=== 3. PCA on standardised signals ===")
print("explained variance:", pca.explained_variance_ratio_.round(2)[:6])
print(pd.DataFrame(pca.components_[:3].T, index=SIGNALS, columns=["PC1", "PC2", "PC3"]).round(2), "\n")

print("=== 4. Congestion slope (extra taxi-out minutes, empty hour -> full hour) ===")
show = ["code", "passengers", "congestion_slope", "slope_r2", "taxi_in_median", "load_factor", "peak_saturation", "nas_share"]
top = df.sort_values("congestion_slope", ascending=False)
print(top[show].head(15).round(3))
print("...bottom 5")
print(top[show].tail(5).round(3), "\n")

print("=== 5. Each signal alone vs congestion slope (Spearman rho, OLS R2) ===")
rows = []
for s in SIGNALS:
    beta, p, r2, n = ols(df.congestion_slope, z[[s]])
    rows.append((s, round(df[s].corr(df.congestion_slope, method="spearman"), 2), round(r2, 2), round(p[1], 3),
                 "circular" if s in TAXI_OUT_BASED else ""))
print(pd.DataFrame(rows, columns=["signal", "rho", "R2", "p", "note"]).sort_values("rho", ascending=False).to_string(index=False), "\n")

print("=== 6. Multiple OLS: congestion_slope ~ standardised signals (robust p-values) ===")
models = {
    "current 5 without taxi-out": [s for s in CURRENT if s not in TAXI_OUT_BASED],
    "all 13 without taxi-out": [s for s in SIGNALS if s not in TAXI_OUT_BASED],
}
for name, preds in models.items():
    beta, p, r2, n = ols(df.congestion_slope, z[preds])
    k = len(preds)
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    print(f"--- {name}: n={n}, R2={r2:.2f}, adj R2={adj:.2f}")
    print(pd.DataFrame({"coef_std": beta[1:], "p": p[1:]}, index=preds).round(3).sort_values("coef_std", ascending=False))
    print()

print("=== 7. Reduced model (one signal per redundant group, no taxi-out) + bootstrap ===")
REDUCED = ["peak_saturation", "load_factor", "delay15_rate", "taxi_in_median"]
beta, p, r2, n = ols(df.congestion_slope, z[REDUCED])
print(f"n={n}, R2={r2:.2f}")
rng = np.random.default_rng(0)
boot = []
for _ in range(1000):
    i = rng.integers(0, len(df), len(df))
    b, *_ = ols(df.congestion_slope.values[i], z[REDUCED].values[i])
    boot.append(b[1:])
boot = np.array(boot)
w_pos = np.clip(beta[1:], 0, None)
res = pd.DataFrame({"coef_std": beta[1:], "p": p[1:],
                    "boot_5%": np.percentile(boot, 5, axis=0), "boot_95%": np.percentile(boot, 95, axis=0),
                    "share_boot_positive": (boot > 0).mean(axis=0),
                    "implied_weight": w_pos / w_pos.sum()}, index=REDUCED).round(3)
print(res, "\n")

print("=== 8. Does the weighting change the ranking? ===")
def norm01(s):
    lo, hi = s.quantile(0.1), s.quantile(0.9)
    return ((s - lo) / (hi - lo)).clip(0, 1)

CAND = ["load_factor", "peak_saturation", "delay15_rate", "taxi_in_median", "taxi_out_median"]
N = pd.DataFrame({s: norm01(df[s]) for s in CAND + ["nas_share", "gauge_vs_flights"]})
schemes = {
    "plan today (5 signals, 0.25/0.25/0.2/0.15/0.15)":
        {"load_factor": .25, "peak_saturation": .25, "nas_share": .20, "taxi_out_median": .15, "gauge_vs_flights": .15},
    "equal, 5 recommended signals": {s: 0.2 for s in CAND},
    "regression-implied + taxi-out": {**{s: 0.8 * w for s, w in zip(REDUCED, w_pos / w_pos.sum())}, "taxi_out_median": 0.2},
}
C = pd.DataFrame({k: sum(N[s] * w for s, w in v.items()) for k, v in schemes.items()})
C["slope"] = df.congestion_slope
print("Spearman between C under each scheme and the congestion slope / each other:")
print(C.corr(method="spearman").round(2), "\n")
C["code"] = df.code
for k in schemes:
    C["rank_" + k[:5]] = C[k].rank(ascending=False).astype(int)
print("Top 15 by 'equal, 5 recommended signals':")
print(C.sort_values("equal, 5 recommended signals", ascending=False)
      [["code"] + list(schemes) + ["slope"]].head(15).round(2).to_string())
