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
explicit straight-sprint reference time. Gate COD gives segment times only.

## Verification

`tests/test_sprint_cod_assessment.py`: noiseless and seeded noisy recovery with
uncertainty coverage, short-window and over-parameterised refusals, start-definition
sensitivity, milestone times against the exact position inverse, gate-split distance fit, a
closed-form piecewise COD profile, refusals for gates, the native radar → engine → F-V
pipeline through the real CLI, and the import boundary. Engineering calculations only.
