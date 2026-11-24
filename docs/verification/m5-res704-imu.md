# RES-704 IMU method admission

The original `imu-frame-observability@1` manifest remains a development episode.
Its 90-degree yaw maps the calibrated sensor-X shear effect to field Y. The
original requested X/Z and terminal X velocity operands therefore give the same
answer when that correction is omitted. Revision 2 preserves all eight artifact
bytes, digests, checkpoint IDs and scenarios. It additionally requests field-Y
free acceleration and peak corrected angular rate, and records each method's
checkpoint assignment in the manifest. The quantity estimands are unchanged.

| Role | Exact dependency and required evidence |
| --- | --- |
| A | One native synthetic IMU, canonical time, sensor-frame XYZ specific force and XYZ angular rate, tied to source and declared axes |
| B | Divide by scale, subtract bias, invert misalignment, and use the declared field-from-sensor orientation for position |
| C | 20 Hz explicit native timestamps, declared identity alignment, two exact clock ties spanning the episode, and unresolved-gap refusal |
| D | Ranges on the untouched raw axes, clipping rejection, and finite ordered native samples |
| E | Gyro propagation with corrected zero rate, no filtering or inferred attitude, and the declared threshold applied to calibrated specific force |
| F | Separate sensor specific force and field free acceleration, then integrate with declared initial velocity and position |
| G | Exactly one contiguous below-threshold run with complete outer brackets and the declared minimum signature duration |
| H | Field-X instrument-point position with explicit observability refusals, separate from the sensor threshold signature duration |
| I | Conditional position radius bT²/2 and conservative sum of both native edge intervals for signature duration |
| J | Independent closed form, separately authored private answers and oracle, and independent derivation and review |
| K | Positive, negative, adversarial and refusal controls for both methods, including rail and native-clock faults |
| M | Exact task, per-checkpoint method, source channel and artifact digests, implementation and reviewed private reference bindings |
| O | DEV-002 interpretation limits, with separate instrument point, whole-athlete COM and physical flight estimands |

L is inapplicable to the single-IMU scenario. Existing supported B, D, E, F, I,
J, K and O receipts support the overlapping device dependencies. The exact
M5-only A/C/G/H/M receipts and tests bind both procedures and add the required
signature-specific G evidence. The existing I receipt covers the position
bias bound and the scoped G receipt directly covers the signature edge-sum
bound. No full-M3 ledger promotion is requested.

The field is ENU and gravity is −9.80665 m/s² along field Z. Specific force is
`Rᵀ(a−g)`. The fixed-attitude method requires a declared proper orientation,
mounting identity under the synthetic placement declaration, calibrated gyro
consistent with zero, constant field acceleration, and both initial conditions.
The norm alone cannot identify tilt or static motion. The c02 observed force is
compatible with both tilted constant-velocity motion and accelerating motion with a different yaw
with zero angular rate. This establishes an observability refusal.

The signature spans the last at/above-threshold sample before the run and the
first after it. The sum of the two edge intervals is a conditional conservative
bound on the outer sample span under the crossing-bracket hypothesis. The
idealized identity clock is assumed exact for the quoted radii. Its calibration
and metadata tolerances are numerical and supply no hardware timing accuracy.
Neither the signature nor the instrument position identifies force-platform
events, physical whole-body flight, jump height, or the whole-athlete COM.

The prespecified absolute grading tolerance is 1e-9 with zero relative tolerance
for each method's quantity, radius and named numeric operands. The required
method per checkpoint, unit, estimand, trial and evidence identities, refusal
decision, citations and interpretation claims must match. Operand order is free.
The oracle must pass every checkpoint. Grading and admission controls require
zero false accepts or false rejects. All four requested baselines run twice
under the fixed campaign seed, and their actual scores are recorded. Deterministic
replay, isolation, exact digest verification, and complete cleanup are required.
