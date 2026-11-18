# RES-670 numerical verification

Generated from `tests/test_force_platform_verification.py`; no empirical device validity is asserted.

| Scenario | Filter | Max velocity error (m/s) | Max displacement error (m) |
| -- | -- | -- | -- |
| cmj | baseline | 0.012521 | 0.007751 |
| cmj | butterworth | 0.055255 | 0.007751 |
| cmj | chebyshev | 0.036523 | 0.007751 |
| sj | baseline | 0.008172 | 0.003915 |
| sj | butterworth | 0.056203 | 0.003915 |
| sj | chebyshev | 0.038677 | 0.003915 |
| dj | baseline | 0.044688 | 0.036761 |
| dj | butterworth | 0.114515 | 0.036778 |
| dj | chebyshev | 0.084616 | 0.036499 |
| rj_10_5 | baseline | 0.031624 | 0.045940 |
| rj_10_5 | butterworth | 0.097242 | 0.045933 |
| rj_10_5 | chebyshev | 0.073568 | 0.045901 |

252 rows cover both plate subfamilies at 600, 1000 and 2000 Hz.
Full method/onset/contact/start sensitivity values: [force-platform-verification.csv](force-platform-verification.csv).
Baseline uses raw force and trapezoid integration. Filtered results include boundary smoothing bias.
The separate exhaustive tests cross all filter/integration/onset choices on every scenario and rate.
Ideal flat analytical plateaus trigger the pre-existing QC-S03 clipping heuristic; these are diagnostic numerical results, not accepted empirical trials.
No pre-movement onset exists in the recorded DJ/hop fixtures; those rows use separate quiet weighing and explicit initial velocity. An onset-start reconstruction is inapplicable there.
All reported errors are against independent RES-669 closed-form trajectories, not vendor outputs or values calculated by the production engine.
These conditional sensitivity tables are not confidence intervals or hardware uncertainty budgets.
