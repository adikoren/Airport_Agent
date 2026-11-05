# Which capacity-constraint signals, and what weights? — EDA findings

Script: [eda_constraint_weights.py](eda_constraint_weights.py) · full output: [output/report.txt](output/report.txt) · per-airport table: [output/airport_signals.csv](output/airport_signals.csv)

Data: BTS T-100 + On-Time, Aug 2025 – Jul 2026; the 12 months before are used for YoY (year over year) comparisons. **105 US airports with ≥ 1M passengers**, 6.46M departures.

## 1. Method

There is no label that says "this airport is capacity-constrained", so we built one from queueing theory: **near capacity, delay rises steeply with each extra flight**.

- For each airport, every (day, hour) cell gets a **utilization** = scheduled departures in that hour ÷ the airport's own 99th-percentile hourly departures.
- Per airport, a weighted regression of mean taxi-out on utilization. The slope = **congestion sensitivity** (extra taxi-out minutes going from an empty hour to a full hour).
- Then we test which of **13 candidate signals** predict that slope: correlations, single-signal fits, multiple regression, PCA, bootstrap.

Face check: the airports with the steepest slope include PHX, PHL, SAN, DCA, SNA, SFO, SEA, ORD and LGA, which are widely known to be runway- or gate-limited. That gives some confidence the outcome measures something real.

## 2. Candidate signals

| Group | Signal | Source |
|---|---|---|
| Full planes | load factor; months with LF > 85% | T-100 |
| Peak usage | peak saturation (mean daily peak hour ÷ p99 hour); share of hours ≥ 90% of p99 | On-time |
| Ground congestion | taxi-out median; taxi-out p90 | On-time |
| **Gate shortage (new)** | **taxi-in median** (waiting for a gate after landing) | On-time (arrivals) |
| System delay | NAS share of delay minutes; NAS minutes per departure; NAS-caused cancellation rate | On-time |
| General delay | share of departures ≥ 15 min late | On-time |
| Airline behavior | bigger planes instead of more flights (gauge growth − flight growth); passengers growth − seats growth | T-100 |

## 3. Results

**Near-duplicates (Spearman ρ > 0.7), so keep one per group:**
- peak saturation ↔ hours at capacity (0.84)
- NAS share ↔ NAS minutes per departure (0.78) ↔ NAS cancellations (0.72)
- taxi-out median ↔ taxi-out p90 (0.74)
- load factor ↔ months with LF > 85% (0.73)

**Each signal alone vs congestion slope:**

| Signal | ρ | R² | Verdict |
|---|---|---|---|
| peak saturation | 0.43 | 0.12 | strong |
| delay ≥ 15 rate | 0.39 | 0.13 | strong |
| taxi-out median | 0.40 | 0.27 | strong, but partly circular (the outcome is built from taxi-out) |
| taxi-in median | 0.23 | 0.10 | moderate |
| load factor | 0.21 | 0.05 | weak alone, significant together with others |
| **NAS share** | **0.04** | **0.00** | **no relation** |
| **gauge vs flights** | **0.10** | **0.00** | **no relation** |
| passengers vs seats growth, NAS min/dep, NAS cancellations, months LF > 85% | ≈ 0 | ≈ 0 | no relation |

**Reduced model** (one signal per group, taxi-out left out because it's circular): R² = 0.31, n = 105.

| Signal | Std. coef | p | Bootstrap 90% interval | Implied weight |
|---|---|---|---|---|
| peak saturation | 1.09 | < 0.001 | 0.61 – 1.55 | 0.38 |
| delay ≥ 15 rate | 0.85 | 0.001 | 0.52 – 1.34 | 0.30 |
| load factor | 0.71 | 0.006 | 0.33 – 1.17 | 0.25 |
| taxi-in median | 0.20 | 0.47 | −0.43 – 0.62 | 0.07 |

**Does the weighting change the ranking?** Spearman rank correlation of `C`:

| | vs congestion slope | vs equal weights |
|---|---|---|
| Plan today (LF, peak, NAS, taxi-out, gauge; 0.25/0.25/0.2/0.15/0.15) | 0.47 | 0.75 |
| **Equal 0.2 × (LF, peak, delay15, taxi-out, taxi-in)** | **0.50** | 1.00 |
| Regression-implied weights + taxi-out | 0.56 | 0.95 |

## 4. What this means

1. **Choosing the signals matters more than the weights.** Equal weights and regression weights rank airports almost the same (0.95). Swapping out the two dead signals changes the ranking far more (0.75).
2. **Drop NAS share and gauge-vs-flights.** Neither relates to congestion. A likely reason for NAS: it's recorded on the departing flight, but much of it is caused at the *destination* airport, so it doesn't measure the origin airport.
3. **Add the departure delay rate (≥ 15 min).** It's the second-strongest signal.
4. **Taxi-in is statistically weak here,** but it's the only gate-side (terminal) signal we have, and the investment is in terminals. Keeping it is a judgment call to state openly.
5. **Spill is ≈ 0 for most airports.** Monthly load factor goes above 85% in 0 months for the median airport (p90 = 3 months), and yearly LF only ranges 0.74–0.82. So the `spill_now` term in RPS step 4 adds almost nothing, and the 0.85 threshold needs rethinking.

## 5. Limits (say these in the design doc)
- The outcome is noisy: utilization explains little of the hourly taxi-out variation inside one airport (median R² 0.006), because weather and runway configuration dominate. The slope is still estimated from ~2,000–6,000 hourly cells per airport.
- n = 105 airports, one year, domestic flights of reporting carriers only.
- Peak saturation and the outcome both use the airport's own p99 hour, so part of their link may be mechanical.
- The delay ≥ 15 rate also includes airline-caused delays, not only airport capacity.
- Taxi-out is in both the outcome and a signal, so it was left out of the fitted models.

---

# Part 2 — Re-check against runway AND gate congestion

Script: [eda_gate_side.py](eda_gate_side.py) · output: [output/report_gate_side.txt](output/report_gate_side.txt)

Part 1 only measured **runway-side** congestion (taxi-out). Gates are inside the terminal, and the terminal is the investment, so part 2 adds a **gate-side** measure. `C` stays **one index**.

## New measures
- **Gate slope**: per airport, how much taxi-in rises from a quiet arrival hour to a full one. Steep = planes wait for a gate.
- **Cleaned taxi-in (`taxi_in_excess`)**: taxi-in in busy arrival hours minus taxi-in in quiet hours, at the same airport. Raw taxi-in mixes gate waiting with airfield size; this keeps only the waiting.
- **Combined congestion**: the average of an airport's rank on the runway slope and its rank on the gate slope.

## Results
1. **Runway and gate congestion are unrelated** (ρ = 0.03). They are two different problems, and the original signals only measured the runway one.
2. **Gate-slope leaders** are big hubs (ORD, ATL, LAX, FLL, PHX, SEA, CLT, MCO) **and small seasonal leisure airports** (BZN, MYR, AZA, VPS), whose few gates are overwhelmed in peak season. That's plausible and relevant for terminal investment.
3. **Signals vs each measure** (Spearman ρ; * = built from the same data, so partly circular):

| Signal | Runway | Gate | Combined |
|---|---|---|---|
| cleaned taxi-in | +0.00 | +0.91* | +0.63* |
| taxi-out median | +0.40* | +0.13 | +0.36* |
| load factor | +0.21 | +0.23 | +0.30 |
| delay ≥ 15 rate | +0.39 | −0.00 | +0.26 |
| raw taxi-in | +0.23 | +0.18* | +0.26* |
| peak saturation | +0.43 | −0.06 | +0.24 |
| NAS share | +0.04 | −0.10 | −0.04 |
| gauge vs flights | +0.10 | −0.15 | −0.05 |

   Load factor is the only non-taxi signal that relates to both sides.

4. **Weighting schemes** (Spearman between each `C` ranking and the measured congestion; bootstrap 90% interval in brackets):

| Scheme | Runway | Gate | Combined |
|---|---|---|---|
| A. Plan today (LF, peak, NAS, taxi-out, gauge; 0.25/0.25/0.2/0.15/0.15) | 0.47 [0.34, 0.59] | −0.01 | 0.30 [0.15, 0.44] |
| B. Equal 0.2, raw taxi-in | 0.50 [0.35, 0.62] | 0.07 | 0.38 [0.24, 0.52] |
| **C. Equal 0.2, cleaned taxi-in** | 0.51 [0.37, 0.63] | **0.34** [0.17, 0.49] | **0.58** [0.45, 0.69] |
| D. Regression weights from part 1 | 0.56 [0.43, 0.68] | 0.01 | 0.38 [0.23, 0.51] |

   Against today's plan, on the combined measure: **C is better by +0.28** (90% interval +0.17 to +0.39; better in 100% of bootstrap samples). **B and D are better by only +0.08 to +0.09**, and their intervals include 0.

## How to read these numbers
- They are **rank correlations**: 1 = `C` orders the airports exactly like the measured congestion, 0 = no relation, 0.5 = a clear but loose agreement.
- Squared, they give roughly the share of the ranking that `C` "explains": 0.47 → 22%, 0.50 → 25%, 0.56 → 31%.
- **0.47, 0.50 and 0.56 are statistically the same.** Their intervals overlap almost completely, so on the runway side the choice of weights doesn't matter.
- Nothing gets close to 1, because the outcome itself is noisy (weather and runway setup move taxi times a lot).

## Conclusion
- Use **one `C` with 5 equally weighted signals: load factor, peak saturation, delay ≥ 15 rate, taxi-out median, cleaned taxi-in.**
- This is the only option that captures gate congestion, which is what a terminal expansion fixes. It is also clearly the best on the combined measure.
- **Caveat:** cleaned taxi-in is built from the same data as the gate measure (ρ 0.91), so its gate advantage is partly built in. The case for it rests on logic (it directly measures waiting for a gate) more than on an independent test.

---

# Part 3 — Cleaned taxi-out and cleaned delay rate

Scripts: [eda_cleaned.py](eda_cleaned.py), [eda_cleaned_check.py](eda_cleaned_check.py) · output: [output/report_cleaned.txt](output/report_cleaned.txt), [output/report_cleaned_check.txt](output/report_cleaned_check.txt)

The same "busy hour minus quiet hour at the same airport" cleaning as taxi-in, applied to taxi-out and to the delay rate.

## Taxi-out: cleaning works
- Cleaned taxi-out removes airfield size. For example, HPN and TUL drop to about 0 (short taxi times are their normal), while ORD (+9.7 min) and LGA (+9.2 min) stay high.
- A within-hour version (same hour of day, busier vs quieter days) agrees well with it (ρ 0.67), so it isn't just a time-of-day effect.
- **Caveat:** it's almost the same thing as the runway congestion measure (ρ 0.94). Its strong match is therefore built in, not independent proof. Like cleaned taxi-in, the case for it is logic: it measures queueing, not distance.

## Delay rate: the simple cleaning fails
- On average, busy hours have **fewer** delays than quiet hours (−4.5 points). That's because the quiet hours are mostly late evening, when the day's delays have piled up. Time of day dominates, not airport busyness.
- The simple version and the within-hour version disagree (ρ 0.28), which confirms it.
- The within-hour version is valid, but it adds nothing measurable (next table).

## Comparison (Spearman with measured congestion; bootstrap 90% interval)

| Option (equal weights) | Runway | Gate | Combined |
|---|---|---|---|
| A. Plan today | 0.47 | −0.01 | 0.30 |
| C. LF, peak, raw delay, raw taxi-out, cleaned taxi-in | 0.51 | 0.34 | 0.58 [0.45, 0.69] |
| G. LF, peak, raw delay, cleaned taxi-out, cleaned taxi-in | 0.70 | 0.32 | 0.70 [0.61, 0.78] |
| H. Same as G, within-hour cleaned delay | 0.68 | 0.41 | 0.76 [0.67, 0.82] |
| **I. LF, peak, cleaned taxi-out, cleaned taxi-in (no delay rate)** | 0.67 | 0.40 | **0.74 [0.65, 0.81]** |

- H and I are not significantly different from G: +0.05 [−0.01, +0.12] and +0.04 [−0.02, +0.09].
- I, H and G rank airports almost the same (0.88–0.94).
- **The delay rate doesn't earn its place**: once both taxi times are cleaned, dropping it changes nothing measurable.

## Conclusion
**`C` = the average of 4 equally weighted signals (0.25 each): load factor, peak saturation, cleaned taxi-out, cleaned taxi-in.**
- **Simplest:** 4 signals, equal weights, one cleaning method for both time signals.
- **As good as any option tested.**
- **Covers both sides:** runway (taxi-out, peak saturation), gate (taxi-in), and demand (load factor).

**Honest caveat:** most of the jump from 0.58 to 0.74 comes from cleaned taxi-out, which is nearly the same as the runway measure. Read the 0.74 as an upper bound, not a validated accuracy. Load factor and peak saturation are the only signals tested against the congestion measures independently.

Cut-offs that are a choice: busy hour ≥ 75% and quiet hour ≤ 25% of the airport's own 99th-percentile hour. They go in config.

---

# Part 4 — How to put the 4 signals on one scale

Script: [eda_scaling_check.py](eda_scaling_check.py) · output: [output/report_scaling_check.txt](output/report_scaling_check.txt)

The 4 signals have different units, so they need a common 0–1 scale before they are averaged into `C`. Three methods were tested on the same 105 airports:

| Method | Values stuck at exactly 0 or 1 | `C` vs combined congestion (ρ) |
|---|---|---|
| 10th–90th percentile, capped (used in parts 1–3) | 88 of 420 (21%) | 0.74 |
| Min–max, no caps | 8 of 420 | 0.64 |
| **Percentile rank** | 4 of 420 | **0.79** [0.72, 0.85] |

- **Min–max is distorted by single extreme airports.** ORD's busy-hour taxi-in is 10.53 extra minutes, while the next airport (BZN) is 5.36. Every other airport is squeezed near 0, and the median scaled taxi-in value is 0.17.
- **Percentile rank gives every signal the same spread** (median 0.5 for all four), so equal weights really mean equal influence.
- Percentile rank beats the capped method by **+0.05** (90% interval +0.02 to +0.08, better in 100% of bootstrap samples).

**Decision:** percentile rank, against a fixed reference of all US airports that pass the hard data checks (ARCHITECTURE §6.3, PLAN D6).

---

# Part 5 — Why there is no "passengers turned away today" (spill) term

An earlier draft of RPS step 4 added today's turned-away demand: `spill_now = Σ seats_m × max(0, LF_m − 0.85)`, the seats above 85% full in each month. It was removed.

1. **Turned-away passengers can't be counted from public data.**
   - T-100 records only passengers who flew. A sold-out flight shows 180 passengers whether 181 or 300 people wanted a seat.
   - Airlines see the unmet requests in their booking systems, but that data is private.
   - Public data shows only indirect traces, such as full planes (load factor) or high fares, never a count.
2. **Counting them needs a "full" threshold, and no threshold can be justified from our data.**
   - Monthly averages never reach 100%, even when many flights sell out: a month mixes full peak flights with emptier off-peak ones.
   - The highest monthly load factor in the data is 92%, so any threshold is a guess about how much empty space the average hides.
   - Real spill needs flight-by-flight booking data.
3. **The result depended almost entirely on that guess** (105 airports ≥ 1M passengers, last 12 months; 1,260 monthly values, none missing):

   | "Full" threshold | Airports with spill > 0 | Median spill, % of passengers | 90th percentile |
   |---|---|---|---|
   | 85% | 31 | 0.0% | 0.4% |
   | 82% | 71 | 0.2% | 1.6% |
   | 80% | 92 | 0.9% | 3.0% |

   At 85%, 74 of 105 airports had zero spill, including busy BOS, whose fullest months reach exactly 85% (monthly values 0.74–0.85). That's a real zero (no month above the line), not missing data, but it reflects the threshold more than the airport.
4. **Ranking doesn't need a count.** It only needs to know which airports have fuller planes than others. Load factor is one of the four signals in `C`, scaled by percentile rank, so no "full" line is needed.

**Result:** RPS uses forecast growth × `C` with no spill term (a later revision also dropped the "unlocked passengers" count, since it needed `C` to be a true share; ARCHITECTURE §6.2). The general rule: **don't add a parameter the data can't support.** The same rule removed the international-passenger weight and the numeric data-quality score.

---

# Part 6 — Are the chosen cut-offs safe? (sensitivity test)

Script: [eda_sensitivity.py](eda_sensitivity.py) · output: [output/report_sensitivity.txt](output/report_sensitivity.txt)

Two choices define three of the four signals in `C`: the busy / quiet hour cut-offs (base 75% / 25%) and the hour used as an airport's "maximum" (base: its 99th-percentile hour). Each was varied, `C` was rebuilt (percentile rank, equal weights), and the ranking was compared with the base. The congestion measure was kept at its base definition as a fixed yardstick.

| Variant | Rank agreement with base | Airports moving > 5 places (of 105) | Largest move | Top-10 overlap | `C` vs congestion |
|---|---|---|---|---|---|
| **Base: 99th percentile, 75 / 25** | 1.000 | 0 | 0 | 10 | 0.79 |
| Busy / quiet 70 / 30 | 0.985 | 24 | 17 | 10 | 0.78 |
| Busy / quiet 65 / 35 | 0.976 | 30 | 25 | 9 | 0.78 |
| Busy / quiet 80 / 20 | 0.984 | 19 | 27 | 9 | 0.76 |
| Busy / quiet 85 / 15 | 0.923 | 49 | 56 | 8 | 0.68 |
| Maximum = 97th percentile | 0.879 | 71 | 52 | 8 | 0.72 |
| Maximum = 95th percentile | 0.810 | 75 | 56 | 9 | 0.64 |
| Maximum = highest hour (100th) | 0.894 | 58 | 36 | 9 | 0.78 |

**Busy / quiet cut-offs: safe.** Anything from 65/35 to 80/20 ranks airports almost the same (≥ 0.976) and keeps 9–10 of the top 10. Only extreme cut-offs (85/15) degrade, because too few hours qualify as busy or quiet.

**"Maximum hour": sensitive.** Changing it moves many airports in the middle of the ranking (up to 75 of 105 move more than 5 places), though the top 10 stays mostly the same (8–9 of 10). The 99th percentile matched measured congestion best (0.79), but the congestion measure itself uses the 99th percentile, so that advantage is partly built in.

**Conclusion:** keep 75/25 and the 99th percentile. State in the design doc that the top of a ranking is stable, while positions in the middle are approximate.
