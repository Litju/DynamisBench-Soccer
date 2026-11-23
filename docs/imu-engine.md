# IMU engine (RES-701)

`dbs imu process <bundle> [--config config.json] [--out report.json]`

Input: schema-v2 IMU bundle with `specific_force_{x,y,z}` (m/s²) and
`angular_velocity_{x,y,z}` (rad/s) in the sensor frame. The configuration holds `spec`
(declarations) and `method` (named variants).

## Estimands are kept apart

| Quantity | Definition | When produced |
| -- | -- | -- |
| specific force | sensed `R^T (a − g)`, sensor (or declared segment) frame | always, after declared calibration |
| free acceleration | `a` in the field frame = `R f + g` | only with an orientation estimate |
| velocity, position | integrals of free acceleration | only with declared initial conditions |

## Calibration, placement and frames

`f = M⁻¹ (raw / scale − bias)` and `ω = raw / scale − bias` from declared records. Without
`calibration_declared` the identity is used and reported as such. A sensor-to-segment
rotation needs a declared placement description. The field frame is ENU with +z up.

## Orientation variants and observability

* `gyro_integration`: exact per-sample Rodrigues propagation of the declared (or static-tilt)
  initial orientation. Drift equals the uncorrected gyroscope bias times time.
* `complementary`: adds a gravity correction `kp (f̂ × R^T ẑ)` when the sensed magnitude is within
  a declared tolerance of g. This gate can also pass during sustained acceleration and
  produce a biased tilt estimate. For small angles under gravity-only conditions, the
  constant gyroscope bias response is approximately `b / kp`.

Initial orientation comes from a declared rotation or a static window (roll and pitch from
gravity, yaw as declared). Neither declared means orientation is **refused**. Heading is never
observable from accelerometer plus gyroscope; no magnetometer is admitted.

## Integrity checks

`IMU-RATE` (rate error), `IMU-GAPS` (missing samples; rejects), `IMU-SYNC` (declared
synchronisation uncertainty), `IMU-SPIKE` (impact artefact), `IMU-SAT-ACCE` / `IMU-SAT-GYRO`
(declared rails; reject), `IMU-GYROBIAS` (static-window gyroscope mean), `IMU-SLIP` (static gravity
direction at the end versus the integrated rotation).

## Events and sensitivity

Flight events from the specific-force magnitude below a declared threshold for a declared
minimum time, with take-off and landing at the bracketing samples (resolution reported) and
the landing impact peak. Sensitivity: flight time at several thresholds, velocity bound `b·T`
and position bound `b·T²/2` for an assumed accelerometer bias, and the tilt leakage rate.

## Sourced limits retained in the report

- The specific-force equation and the independent rigid-body fixture explain
  gravity compensation and why an offset sensor point differs from segment
  translation. For an attitude error θ, the leaked gravity components are
  g·sin θ horizontally and g·(1−cos θ) vertically.
- The orientation and integration definitions above give the yaw
  observability limit, bias growth, the conditional gravity correction, and
  the need for initial conditions. Only ZUPT requires declared zero-velocity windows.
- Registry claim DEV-002 (Nielsen et al.) distinguishes sensor displacement
  from center-of-mass displacement and limits agreement to the tested device,
  placement, and methods. Sensor mounting is an input declaration.
- The flight definition uses sampled specific-force thresholds. The integrity
  contract declares rails and timing uncertainty. It cannot recover clipped
  samples, establish empirical synchronization, or validate a placement.

## Verification

`tests/test_imu_engine.py` compares against RES-669 closed forms (rigid-body rotation and
centripetal acceleration, trajectory flight time, free fall senses zero), derives bias
growth and complementary steady-state error by hand, inverts a forward calibration model,
and injects bias, misalignment, saturation, gaps, spikes, rate error, static gyro bias and
slip. The engine does not import the substrate.

Engineering calculations only; no placement, device or physiological qualification.
Take-off/landing definitions for specific assessments belong to RES-671.
