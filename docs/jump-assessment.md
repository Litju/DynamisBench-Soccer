# Jump assessment (RES-671)

`dbs assess jump <force-plate bundle> --config config.json [--out report.json]`

Config: `jump` (`JumpSpec`), `method` (force-platform engine method) and `plates`. DJ and
hop records without a quiet start need `--weighing` plus `integration_start: record_start`
and an explicit initial velocity, exactly as the force-platform engine requires.

The layer consumes force-platform observations (force, velocity, displacement, body weight)
and adds protocol logic. It does not re-calibrate or re-filter.

## Protocols and eligibility (checked, never assumed)

| Protocol | Eligibility checks |
| -- | -- |
| `cmj` | quiet standing, movement onset, unweighting depth ≥ declared fraction, one complete flight |
| `sj` | quiet standing, onset, unweighting depth ≤ declared fraction, one flight |
| `slcmj` | CMJ checks plus two plates with declared sides and unloaded-plate leakage ≤ limit |
| `dj` | descending initial landing, complete rebound, declared drop height and independently sourced touchdown velocity |
| `repeated_hops` | at least a declared number of complete hop cycles |

An ineligible record still reports what it can, with `eligible: false` and
`diagnostic_only: true`. A device-engine QC rejection is carried through as `device_verdict`.

## Events, phases and estimands

Take-off and landing are the first sample on each side of the contact threshold (resolution
one sample). CMJ/SJ phases come from velocity extrema and zero crossing: unweighting,
braking, propulsion, time to take-off. Quantities: flight time, take-off velocity,
**height from impulse-momentum** and **height from flight time** (kept as separate named
estimands with their difference), unweighting depth, peak force and power, propulsive net
impulse, modified RSI. DJ: ground contact time, rebound height and RSI with a declared height
basis (`flight_time` or `impulse_momentum`; different numbers). Hops: per-hop contact, flight,
height and RSI plus means. Bilateral peak force and impulse need declared plate sides.

## Other devices

`contact_system_estimates` and `imu_estimates` convert those engines' flight results into
labelled flight-time estimands with resolution bounds. `compare_sources` reports pairwise
numerical differences with `merged_value: null` and the statement
`empirical agreement: not established (RES-703)`.

## Uncertainty

Timing resolution (one sample), height resolution `g t dt / 4`, flight time at several contact
thresholds, impulse-momentum height under a declared body-weight error. Not a full budget.

## Verification

`tests/test_jump_assessment.py`: closed forms from the RES-669 phase definitions for CMJ,
SJ, DJ and hops through the real force-platform engine, eligibility in both directions,
negative controls (no flight, non-uniform timing, wrong drop height, swapped sides, two-leg
SL-CMJ), clipping corruption carried to `device_verdict`, cross-device source separation
using the contact and IMU engines, and the CLI. Engineering calculations only.

For DJ CLI runs, provide `drop_evidence` alongside `jump`, with a negative
`touchdown_velocity_m_s` and a `source` identifying how it was established independently
of drop height and force integration. Set `jump.drop_height_m` from the declared protocol.
Force must bracket the initial touchdown with an airborne sample followed by contact.
A record starting in contact cannot establish the start of ground contact.
Use separately justified initial conditions in the force processing method. The assessment
compares touchdown with both the declared drop model and the integrated touchdown state.
Failed eligibility or device rejection places the DJ calculations under `diagnostics`, with
empty accepted estimands and contacts.

The engine-v0.1 snapshot covers the registered method specification schemas and CLI
arguments. The jump CLI configuration envelope carries contextual `drop_evidence`
separately from the frozen `JumpSpec` method declaration.
