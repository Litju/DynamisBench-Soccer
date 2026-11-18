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
