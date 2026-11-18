# RES-674 reliability and individual change

## Intraclass correlation

Call `dynamisbench_soccer.statistics.reliability.icc` with a balanced rectangular
matrix of subjects by ratings, a declared model, a rating type and a missing-data
policy. For two-way models each column identifies the same rater or occasion
for all subjects. The one-way model treats rating slots as exchangeable within
subject. Type `single` estimates reliability of one reading, while `average`
estimates reliability of the mean of all columns.

| Model | Definition | Single / average |
| --- | --- | --- |
| `one_way_random` | Absolute agreement across exchangeable ratings | ICC(1,1) / ICC(1,k) |
| `two_way_random` | Absolute agreement across a shared sample of raters | ICC(2,1) / ICC(2,k) |
| `two_way_mixed` | Consistency for the specified fixed raters | ICC(3,1) / ICC(3,k) |

Let B be the subject mean square, E the residual mean square (pooled
within-subject mean square in the one-way model), J the rater mean square,
and n and k the subject and rating counts. Let c = k for single ratings and
1 for averages. The estimate is (B − E)/(B + D), where D = (c − 1)E and
the two-way random model adds c(J − E)/n to D. Intervals default to 95%.
They invert F limits for the one-way and consistency cases and use the
McGraw–Wong approximate degrees of freedom for two-way absolute agreement.
Negative estimates and bounds are retained. Constant data and mean squares
with an undefined estimate or interval raise an error.

NaN and None denote missing measurements. Declare `raise` to reject them
or `omit` to remove the entire subject row. Infinities always raise an error.
The result records the retained and excluded counts. At least two complete
subjects and two ratings are required. Listwise exclusion can introduce bias
when completeness depends on the measurements.

### Reproduction and numerical falsifiers

Run `uv run pytest tests/test_statistics_icc.py`. The fixed six-target, four-judge
matrix is [Shrout and Fleiss (1979), Table 2](https://doi.org/10.1037/0033-2909.86.2.420),
also distributed as [judges.dta](https://www.stata-press.com/data/r19/judges.dta).
All 18 point and interval values come from examples 1–3 of the
[Stata ICC manual](https://www.stata.com/manuals/ricc.pdf).
The same data must give ICC(2,1) 0.2897638 and ICC(3,1) 0.7148407 with their
distinct intervals. Identical variable ratings must yield 1, and a completely
constant matrix must raise an error. Tests also cover listwise deletion and
an exact negative one-way ICC.

## Paired reliability and agreement

`paired_reliability` and `bland_altman` accept one pair per independent subject.
Column order is baseline/reference followed by retest/comparator. A pair with
a missing member is rejected under `raise` or excluded under `omit`. Both
functions report the number of retained and excluded pairs. Infinities always
raise an error and at least two complete pairs are required.

| Quantity | Definition |
| --- | --- |
| Typical error (TE) | Sample SD of retest minus baseline differences divided by √2. Assumes independent, equally precise readings and stable underlying performance. |
| CV% | 100 × TE / grand arithmetic mean of the two columns. Defined here for strictly positive ratio-scale measurements. This is an additive-error CV. |
| SEM | Baseline between-subject sample SD × √(1 − single-reading ICC). Supply an ICC for the same protocol and cohort and report its model and CI with the analysis. |
| SWC | Baseline between-subject sample SD × multiplier. The default multiplier is 0.2. |
| MDC90 / MDC95 | √2 × SEM × the standard-normal 95th / 97.5th percentile, respectively. Thresholds apply to the difference between two independent single readings under a constant-error model. |
| SNR | SWC / TE. Returns `None` if TE is zero. |
| Bland–Altman bias | Mean comparator minus reference difference. |
| Limits of agreement | Bias ± normal quantile × sample SD of the paired differences. Defaults to 95% coverage for individual differences from the pair population. |

Systematic shifts remain visible in the signed Bland–Altman bias. They contribute
to an absolute-agreement ICC but leave SD-based TE unchanged when the shift is
constant. The normal limits of agreement assume approximately normal differences
and stable error across the measurement range. This implementation gives point
estimates of the limits. The 0.2 SWC is a standardized heuristic whose practical
importance depends on the assessment, population and intended decision.
[Hopkins (2000)](https://doi.org/10.2165/00007256-200030010-00001) ·
[Hopkins (2004)](https://sportsci.sportsci.org/jour/04/wghtests.htm) ·
[Bland and Altman (1986)](https://doi.org/10.1016/S0140-6736(86)90837-8)

## Individual change

`individual_change` takes historical sessions in chronological order and a
separate current session. Declare `BaselinePolicy` with `fixed` (first N
historical sessions) or `rolling` (latest N), a minimum number of baseline
sessions (at least two), a minimum trial count per session and a missing-data
policy. The fixed window stops growing when it reaches N sessions. Select the
window before applying missing-data rules. Under `omit`, drop missing trials
and exclude short history sessions inside the selected window without backfill.
Under `raise`, missing trials or a short selected session fail. An insufficient
baseline or current session fails under either policy. The output records the
original baseline indices, usable trial counts, TE, confidence and policy.

Each session score is the arithmetic mean of its trials. Give each eligible
baseline session equal weight. The reported change is current mean minus the
mean of those scores. The descriptive z-score is change divided by the sample
SD of the historical session scores. A flat baseline returns `None` for the
z-score.

Supply a positive, externally estimated TE for an individual **trial**. Estimate
this under a matching protocol. Let m be the number of baseline sessions, nᵢ the
trial count for baseline session i, and n₀ the current count. Under independent,
equal-variance normal trial errors, the change SE is
`TE × sqrt(1/n₀ + sum(1/nᵢ)/m²)`. The limits are change ± the central normal
critical value × SE. Under a flat prior for the true additive change and a known
error scale, the probability of its observed direction is
`Φ(abs(change)/SE)`. The result is `likely_real` if the central interval
excludes zero in either direction, and `within_noise` otherwise. At zero change
the reported probability is 0.5. Positive and negative changes refer to numeric
direction. Whether either is beneficial or exceeds a worthwhile threshold is a
separate decision.

The interval is conditional on the supplied TE and treats the trial errors as
independent across and within sessions. Shared session effects, serial
correlation, fatigue, learning, drift, changing error variance and uncertainty
in TE can change its coverage. Z-scores also reflect variation in the
historical session means. Repeated decisions have no multiplicity adjustment.
This library establishes numerical behavior on reference inputs. Empirical
reliability, population-specific thresholds and device qualification require
separate studies.

### Reproduction and numerical falsifiers

Run `uv run pytest tests/test_statistics_change.py`. The test fixture uses the
first readings from [Bland and Altman (1986), Table 1](https://jhanley.biostat.mcgill.ca/bios601/Surveys/BlandAltman1986.pdf)
in large-minus-mini order. Published rounded bias and SD are −2.1 and 38.8 L/min.
Python's standard-library `mean`, `stdev` and `NormalDist` independently give
limits [−78.0959055, 73.8606113] L/min at 95% normal coverage. Swapping readings
must reverse bias and limits.

A synthetic paired fixture has exact sample variances 20/3 for differences and
500/3 for baseline scores. With ICC 0.96 and multiplier 0.3, hand reductions and
standard-library normal quantiles give TE 1.825742, CV 7.302967%, SEM 2.581989,
SWC 3.872983, MDC90 6.006156, MDC95 7.156777 and SNR 2.121320.

Individual-change fixtures use the [RES-669 force ramp](reference-substrate.md)
on its analytical force plateau and an independent `NormalDist` calculation.
Baseline means must be 6 N (fixed) and 9 N (rolling) for the same history. At a
current mean of 15 N the rolling result must give z = 2, change = 6 N and 95%
limits [2.7993922, 9.2006078] N. A separate unequal-count fixture checks the
change SE against 5/√12, a zero descriptive spread, and missing-session
exclusion without backfill.
