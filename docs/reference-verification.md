# RES-669 engineering verification evidence

This matrix records implemented mathematical checks. It does not mark device engines
verified or turn simulated observations into empirical evidence. The checked source
and tests live in `src/dynamisbench_soccer/reference/` and `tests/test_reference_*.py`.

| Ledger subfamily | Independent mathematical relation | Serialized/fault evidence |
| -- | -- | -- |
| force_platform.single | m(g+a), impulse-momentum, GRF vs net power, mass scaling | Export roundtrip, baseline/drift/clipping/drop/protocol faults |
| force_platform.dual | Left+right=total; declared partition; swap involution | Per-plate identity/units, sum policy, L/R/protocol negatives |
| lpt_encoder.lpt | Euclidean attachment-to-anchor distance, distinct COM motion | Raw cable lengths, rate refinement, seeded perturbation |
| lpt_encoder.encoder | Nearest count, error <= half-count; integral samples | Raw counts, missing/duplicate/order records |
| load_cell_isometric.load_cell | V=0.001F+0.2; analytic force ramp/RFD/impulse | V units/inverse calibration, seeded noise |
| load_cell_isometric.isometric_rig | Prescribed rigid-rig force and lever | Calibration/source records, noise and wrong-input controls |
| nordic_hip.nordic | Perpendicular torque/force; explicit actuator torque/work closure | Bilateral forces, breakpoint/endpoints, noise |
| nordic_hip.hip | Isometric force, declared protocol position/action | Hip geometry/calibration, source independence, noise |
| timing_contact.gates | Spherical marker/beam intersection; exponential position inverse | Missing/duplicate/order/latency and wrong-beam negatives |
| timing_contact.contact | Paired flight boundaries; separate receipt latency | Binary events; missing/duplicate/order and double-contact/step-off |
| radar_laser.radar | LOS speed=cos(angle)*truth velocity | Native clock, clipping/offset/drift/noise/resampling |
| radar_laser.laser | Range=truth position+declared initial range | Geometry/native clock/seeded noise |
| gnss_lps.gnss | ENU/ECEF/WGS84, NGA axes, independent PROJ example | Native-rate fixes, latency, metre-space jumps/noise, dropouts/interpolation |
| gnss_lps.lps | Common exact sensor point in declared field axes | Separate native rate, drift/offset mapping, metre-space faults |
| imu.inertial | R.T(a-g), body gyro, proper R, sensor offset derivatives | Complete xyz frames, full orientation, bias/misalignment/saturation |

## Inspectable tests

- `test_reference_complete_scenarios.py`: complete-flight pairing, hop momentum and
  endpoints, net/GRF energy relations, explicit Nordic torque/work closure, arbitrary-axis
  rotation/derivative/IMU checks, known WGS84 axes and published PROJ coordinate example,
  synchronization inverses, beam-height negatives, trapezoid convergence to exponential
  distance, invalid mechanical domains.
- `test_reference_exporters.py`: exact 15-subfamily coverage, every Parquet/JSON
  roundtrip and repeated-write equality, independent per-device relations, all supported
  vertical variants, separate native-rate GNSS/LPS/IMU mapping to common truth, assessment
  conditions, common scenario digest, all-subfamily rate refinement and mass metamorphism,
  impossible exports, and mutation isolation of metadata/arrays.
- `test_reference_faults.py`: exact injected effects for every fault family, deterministic
  seeded content/serialized bytes across all 15 subfamilies, parameter/paradigm/no-op
  negatives, stage refusal, parent integrity, metadata/array isolation, applicable QC
  detection, contact state and L/R metamorphic controls.
- `test_reference_acceptance.py`: frozen seeded hashes for six actual artifact files,
  complete A–O shape/ownership, J/K source/test bindings, preserved qualification gate.
- Earlier `test_reference_kinematics.py`, `test_reference_observations.py`,
  `test_reference_spatial.py` and `test_reference_spatial_observations.py` retain their
  existing mathematical invariants and the two-way engine/reference import boundary.

## Tolerances and numerical independence

Independent fixed cases and algebra usually use 1e-12 absolute tolerance. Gate inverse
positions are checked to 1e-11 m for the tested ranges, after 96 bisection iterations;
no engine's event detector supplies the target. Rigid-body finite differences use
1e-4 s increments and 1e-8 absolute comparison for velocity/acceleration. The reviewer
also independently checks arbitrary initial rotations and body axes.

ENU/geodetic roundtrips are checked within 3e-9 m on the tested local cases; the
published PROJ topocentric example is checked within 1 mm, accommodating its printed
precision and GRS80/WGS84 ellipsoid difference. Latitude inversion has a 20-iteration
limit and 1e-15 rad convergence criterion, with explicit failure outside supported
near-surface cases. Quantization has an exact half-count bound, not a statistical error
claim. Seeded byte pins require the frozen Linux lockfile; library changes that alter
them require review. The existing v1 Parquet golden hash is retained independently.

Trapezoidal integration of the smooth sprint velocity is compared to its closed-form
position at three refinements; the error ratio must approach four. This numerical
method is separate from the truth expression. Discontinuous phase-boundary derivatives
are tested by one-sided/paired physical invariants, not a false smoothness assumption.
No test computes its truth by calling a production processing/assessment engine.

J/K remain partial because this issue supplies the substrate, while verification of
actual device engines against it has not occurred. No other A–O rating or agreement
verdict changes. Review and exact-head CI evidence are recorded on PR #5 and RES-669;
scientific expert qualification is not implied by independent agent code review.
