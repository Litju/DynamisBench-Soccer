# Sprint and COD assessment (RES-673)

* `dbs assess sprint-fv <velocity-result.json> --config config.json [--out report.json]`
* `dbs assess cod <velocity-result.json> --config config.json [--out report.json]`

Input is the JSON result of a velocity engine (`dbs radar-laser process`, GNSS/LPS tracks)
that carries `time_from_start_s` and `velocity_m_s`. Timing-gate splits go through the Python
API (`fit_split_times`, `assess_gates`).

## Sources keep their own estimands

| Source | Estimand | Instantaneous velocity |
| -- | -- | -- |
| radar | Doppler radial speed, declared fixed-angle correction | yes |
| laser | range-rate of laser distance to the declared aiming point | yes |
| GNSS Doppler | Doppler speed over ground | yes |
| GNSS / LPS position | horizontal speed differentiated from positions | yes |
| gates | segment-average velocity between beam crossings | **no, refused** |

No pooled speed exists. A gate source can never feed a speed fit or a COD speed analysis.

## Sprint quantities

Start definitions: `trace_origin`, `velocity_threshold` (interpolated crossing),
`external_trigger`. Milestone times integrate the labelled speed from the declared start.
The F-V model is mono-exponential `v(t) = Vmax (1 − exp(−(t − t0)/τ))` on flat ground with
zero air drag (net horizontal force): `F0 = m Vmax/τ`, `Pmax = F0 Vmax/4`, slope `−F0/Vmax`.
A distance-time variant fits gate splits without differentiating them.

Fit diagnostics: RMSE, R², parameter standard errors from the local covariance, the vmax/τ
correlation, scaled-Jacobian condition number and the window length in time constants.
Parameters are **refused** (not stabilised) when the window covers fewer than a declared
number of time constants, the fit is ill-conditioned or too correlated, the relative standard
error is too large, the optimizer fails or the points cannot identify the parameters.
`fv_start_sensitivity` refits under alternative start definitions.

## COD

A declared path is mandatory: description, length, turn distance and angle. Trace metrics:
total time, turn time and position (minimum speed in the turn window), entry, minimum and exit
peak speeds, deceleration and re-acceleration times, speed loss. The COD deficit needs an
explicit straight-sprint reference time. Trace COD interpolates velocity at the declared
start, integrates piecewise-linear speed, and solves for distance crossings within each
sample interval. Course metrics stop at the declared finish. Gate COD requires cumulative
path positions increasing from zero to the full path length and elapsed times increasing from
zero. It reports segment times and average speeds.

## Verification

`tests/test_sprint_cod_assessment.py`: noiseless and seeded noisy recovery with
uncertainty coverage, short-window and over-parameterised refusals, start-definition
sensitivity, milestone times against the exact position inverse, gate-split distance fit, a
closed-form piecewise COD profile, refusals for gates, the native radar → engine → F-V
pipeline through the real CLI, and the import boundary. Engineering calculations only.

## Source-bound engine handoff (RES-673 correction)

`dbs assess sprint-fv` and `dbs assess cod` read engine JSON through
`assessment/handoff.py`. The instrument is identified from the report, not from the caller:

| Report | Identified as | Speed used |
| --- | --- | --- |
| radar/laser (`native_quantity` velocity / distance) | `radar` / `laser` | `velocity_m_s` on `time_from_start_s` |
| GNSS (`acquisition.device`) | `gnss_doppler` or `gnss_position` | `doppler_speed_m_s` or `horizontal_positional_speed_m_s` on `clock.elapsed_s` |
| LPS | `lps_position` | `horizontal_positional_speed_m_s` on `clock.elapsed_s` |

A declared `source` that disagrees with the report is refused. Non-native input (`manufacturer_processed`,
`vendor_processed`) is refused unless the configuration sets `allow_vendor_processed`, and is then
labelled in the output. Reports carrying a reject/failed verdict or `diagnostic_only` are refused. A trace
with speed below -0.05 m/s (the radar/laser engine's sign bound) is refused; `abs()` is never applied. Samples in
[-0.05, 0) are set to zero explicitly and counted in `input.clamped_sub_tolerance_negative_samples`. GNSS Doppler
needs `doppler_basis = horizontal`; its basis, timestamp basis and latency are carried in the provenance.
The current engines never emit a verdict (they refuse by raising), so the verdict check protects only against
hand-edited or future reports. The output carries an `input` record with source,
device, stage, estimand, clock origin and alignment uncertainty, and engine provenance.

Optional `milestone_distances_m` reports the first physical arrival at each distance from the exact
(fractional) declared start. Speed is piecewise linear and inverted exactly inside each interval;
a stationary tail never moves an arrival and distances beyond the observed support return `null`.

Not supported, and refused instead of invented: gate sources for speed fits, and 505, 30-15 IFT and RSA
protocols as scored tests (only declared-path COD on a speed trace or gate segment times exist).
Tests: `tests/test_assessment_handoff.py`, `tests/test_sprint_milestones.py`.
