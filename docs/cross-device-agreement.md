# Cross-device agreement (RES-703)

`dbs agreement run [--out report.json]` runs every analytical study.
The committed result is `docs/verification/cross-device-agreement.json`; a test regenerates it
and compares.

## Five evidence states, kept apart

| State | Meaning |
| -- | -- |
| `analytical_model_agreement` | idealised observation models of one truth agree within stated resolution; says nothing about real devices |
| `empirical_device_agreement` | measured, independent, paired observations (at least 20) from independent subjects; none exist yet |
| `definition_mismatch` | the estimands differ by definition; the numerical difference is diagnostic only |
| `unsupported_comparison` | one side cannot be compared (for example no vendor-corrected data or algorithm) |
| `missing_evidence` | no paired data, or an empirical claim without measured data |

`classify` never returns "interchangeable"; the only outputs are `unknown` and `no`. Synthetic
and analytical bundles are never counted as measured (`is_measured`).

## Statistics

Paired bias, SD of differences, typical error (SD/√2), normal-theory limits of agreement,
proportional bias (regression of the difference on the mean with SE and p), ratio bias,
expanded uncertainty of the difference from declared standard uncertainties, and a declared
tolerance check that requires the bias *and* both limits to lie within the tolerance.
Undefined quantities are `null`, never NaN.

## Studies (all analytical)

| Comparison | Result class | What differs |
| -- | -- | -- |
| force plate vs contact system, flight time | analytical agreement | switch latency and 1 ms clock resolution |
| force plate vs IMU, flight time | analytical agreement | specific-force threshold and rate |
| force plate vs load cell, isometric force | analytical agreement | rig compliance and alignment are not modelled |
| radar vs gates, split time | analytical agreement | start offset adds to every split |
| radar vs laser | definition mismatch | Doppler velocity vs differentiated range; five-point derivative noise = √130/12 · σ/dt |
| radar vs GNSS and LPS, running speed | definition mismatch | positional latency bias = Vmax e^(−t/τ)(e^(L/τ) − 1) |
| radar vs gates, sprint velocity | definition mismatch | segment average over distance vs speed at the time midpoint |
| force plate vs LPT, jump height | definition mismatch | start-referenced displacement vs flight rise |
| force plate vs LPT, cable vs vertical | definition mismatch | tilted cable under-reads vertical rise |
| force plate vs LPT, attachment vs COM power | definition mismatch | system mass and attachment velocity |
| Nordic raw vs vendor-corrected | unsupported | no data or algorithm; the engine refuses to pool or convert |

## What is not claimed

No comparison is upgraded from `unknown`. Ledger entries now carry `evidence_level =
analytical` and point to the JSON report; empirical interchangeability needs measured paired
data, an accepted tolerance and independent review (M8).
