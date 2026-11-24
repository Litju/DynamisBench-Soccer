# RES-704 LPT reference admission

The scope is the frozen `lpt-vertical-attachment@1` task and its five original
checkpoint IDs and source hashes. The reference uses one 50 Hz native synthetic
cable measuring positive extension in metres, with a source-owned binary slack
channel, a right-handed Z-up field, and a declared constant effective angle from
upward vertical. Identity calibration is a fixture declaration.

| Role | Required task evidence |
| --- | --- |
| A | Canonical native cable distance, timestamp and slack columns and identity calibration on one source |
| B | Signed vertical attachment displacement from relative cable extension and the declared positive direction cosine. Withhold vertical motion when the angle is unknown. |
| C | One explicit uniform 50 Hz native timebase. Validate the whole record, reject repeated or missing ticks and distinguish a common clock scale error from an origin offset. |
| D | Inspect source-owned slack and engine integrity flags before scoring or propagating the trajectory. |
| E | Unfiltered adjacent centered first and second differences at an interior row. Nonquadratic witness controls distinguish the stencils. |
| F | Attachment displacement, speed and acceleration agree with independent closed forms under the selected projection. |
| H | Bar attachment vertical speed in m/s with named displacement, acceleration and conditional load power operands. Keep cable-coordinate and athlete COM estimands distinct. |
| I | Independent speed deviations over joint angle and common clock-scale corners on the declared nonsingular acute interval. |
| J | Independent analytical trajectory and signed/nonquadratic controls, separately authored private answers and an independent oracle |
| K | Inject missing angle, slack, corrupted clock and invalid semantics and verify that no quantity is emitted. |
| M | Bind source and artifact hashes, task identity, implementation, receipt digest, candidate release and sealed answers. |
| O | Conditional rigid translated-load power with declared mass and gravity. DEV-001 contextualizes the instrument scope. |

G and L are inapplicable. The task estimates a pointwise kinematic quantity on
a single instrument and performs no phase detection or device comparison.
Existing C/F/I/J/M receipts cover their overlapping native engine claims.
Scoped A/B/D/E/H/K/O receipts and independent task tests verify the exact
projection and conditional model. The H evidence also verifies the joint
angle/clock corners, beyond the separate perturbations in the existing I receipt.
The M3 ledger and its broader LPT ratings remain outside these scoped receipts.

The task computes relative vertical displacement `z=(L−L₀)/cos(θ)` and
the velocity from adjacent cable samples divided by their elapsed time and
the same direction cosine. The adjacent second difference divided by the
squared sample interval gives vertical acceleration after projection. The
native acquisition engine supplies calibrated length, geometry, time and QC.
The task's local adjacent acceleration stencil is exact for its quadratic
reference. The engine's general double-gradient method also recovers this
quadratic but can differ on nonquadratic motion. Controls on cubic and quartic
motion distinguish the adjacent rule, preserve negative velocity and check
mass-model power with varied mass and timing bounds.

The model power operand is `m(g+a)v` for a declared rigid vertical translation
and `g=9.80665 m/s²`. It pertains to the prescribed moving load, given the
stated geometry, acceleration and velocity. Measured force-platform power,
whole-athlete COM power and real cable geometry lie outside this scope.

Conditional speed uncertainty uses the largest absolute deviation over the
four joint angle and uniform clock-scale corners. The angle interval is acute
and nonsingular. The shared clock origin cancels from the sampled differences.
These bounds describe the model under the stated exact synthetic assumptions.

The prespecified acceptance tolerance is 1e-9 absolute, with zero relative
tolerance for quantities, radii and numeric evidence. Private replay uses
the same canonical 10-decimal operands published by this implementation.
Unit, estimand, method, citations, claims and evidence identities must match.
All five independently authored oracle decisions must score correctly.
Positive, negative, adversarial, admission and replay controls must match
their prespecified decisions with zero false accepts or rejects. Record all
four requested baselines and their observed scores, private-answer isolation,
released inputs, provenance and complete workspace cleanup.
