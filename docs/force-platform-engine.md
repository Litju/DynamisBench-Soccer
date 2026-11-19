# RES-670 production force-platform engine

`dynamisbench_soccer.engines.force_platform.process` performs actual single/dual
force-platform processing. The production package never imports the analytical
reference package. Independent tests supply RES-669 observations and compare outputs
to closed-form motion, not vendor results. RES-671/672 retain assessment-specific
phase definitions, interpretation and protocol-qualified metrics.

## Run acquisition files

```sh
uv run dbs force-platform process acquisition.csv --csv --config method.json \
  --out result.json --traces traces.csv
```

A generic dual-plate CSV with unrecorded coordinate frames/clocks needs explicit
assumptions, rather than a silent claim of verified geometry or synchronization:

```json
{
  "method": {"assume_shared_clock": true},
  "plates": [
    {"plate": 1, "assume_field_axes": true},
    {"plate": 2, "assume_field_axes": true}
  ],
  "assignment_reference": {"1": "left", "2": "right"}
}
```

The assignment must come from an independent plate-to-side record; do not invent it.
Assumed axes/clocks remain explicit QC flags. Schema-v2 bundles with known field
frames and declared clock alignment work without these assumptions:

```sh
uv run dbs force-platform process trial --out result.json --traces traces.csv
```

One or two separate bundle stems are accepted. Separate files must describe the same
subject/session/trial/test/protocol/data category. Different native grids require an
explicit `"alignment": "linear_overlap"`; timestamps must still be uniform within
each device. The overlap uses the first plate's reference-time grid, linear interpolation
and no extrapolation. Gaps/duplicates/out-of-order acquisition times reject.

Exit codes: 0 for accepted/flagged numerical reports, 1 for QC-rejected diagnostic
reports, 2 for invalid inputs/configuration. The JSON always distinguishes diagnostic
results. Rejected diagnostics cannot be treated as qualified measurements. Output
paths cannot overwrite acquisitions/configuration; the report binds the trace CSV's
SHA256. Undefined pre-integration velocity/displacement and unloaded COP are `nan`
in CSV, never manufactured zero measurements. JSON metrics use finite numbers or null.

## Calibration, zeroing, axes and COP

- Already-SI force/torque channels pass through. Upstream calibration/zeroing records
  are retained in provenance, never applied twice.
- Raw voltage requires a full-row-rank, well-conditioned explicit linear matrix,
  ordered input channel names/units and output components `fx,fy,fz,mx,my,mz` as
  applicable. Output forces are N and moments N*m. No proprietary conversion is inferred.
- Raw offsets are subtracted before the matrix. An explicitly designated unloaded
  window or supplied component zero offsets are subtracted afterward, independently
  per plate. Loaded weighing and unloaded zero windows cannot overlap. Zero-window SD
  is reported, not substituted for a calibration uncertainty certificate.
- Static metadata frame chains are composed into a common field frame. An explicit
  `rotation` overrides that mapping. Unknown or mixed measurement frames reject unless
  a field-axis assumption is explicitly enabled; rotating partial force vectors rejects.
- Measured moments are transported from the declared surface origin to the common
  origin. COP uses the actual surface plane normal `n` and point `r0`:
  `(n cross M + (n dot r0) F)/(n dot F)`. A normal-load threshold masks unloaded COP.
  Per-plate and combined COP have field XYZ coordinates. No moments means unavailable
  COP. Dual moment/COP inputs require explicit origins; noncoplanar combined COP is
  unavailable while individual plate COP remains available.

For example, a one-channel raw voltage matrix can declare `channels=["v1"]`,
`input_units=["V"]`, `output_components=["fz"]`, `matrix=[[1000]]` and
`raw_offsets=[0.2]` for the *prescribed example* F=1000(V-0.2). Actual devices require
their own calibration/axes/geometry records. Matrix units, channel order, origins,
zero offsets, raw ADC/output rail limits and axis choices enter the method specification.

## Timebase and reconstruction

Clocks map device timestamps to declared reference seconds using their recorded affine
alignment. Receipt latency is separate and explicitly supplied per plate. Unknown
reference epochs do not certify dual synchronization. The processor uses measured
reference spacing and requires at least 500 reference samples/s; this engineering
minimum is not universal evidence for early RFD or every instrument.

Bodyweight is the mean of a declared >=1 s stable loaded window; mass=BW/g. Invalid
or unstable weighing rejects before integration. A DJ/hop record without preceding
quiet stance requires a separate matching-subject/session/date/category weighing
acquisition and explicit initial velocity at the selected integration start. Impact
force is never used as a quiet-standing bodyweight estimate. Initial velocity is not
inferred from a vendor metric or a drop-height label.

Velocity integrates `(F-BW)/m`; displacement integrates velocity relative to zero at
the selected start. Named starts are `quiet_start`, `onset` and `record_start`.
Both trapezoid and offline cumulative Simpson are implemented. Simpson uses neighboring
quadratic fits, including a future neighbor and the SciPy two-point fallback/tail policy;
it is not a causal streaming estimate. Independent polynomial tests and convergence
checks validate the integration separately from jump fixtures.

## Preserved calculation alternatives

| Choice | Implemented variants and convention |
| -- | -- |
| Force filtering | `raw` default, Butterworth, Chebyshev I; explicit order/cutoff/ripple |
| Filter realization | DC-normalized SOS forward/backward with odd padding; cutoff is the one-pass value, resulting magnitude is squared |
| Onset | `5sd` with explicit N floor; `absolute_N`; sustained high excursion then backward scan to the SD band; filtered first derivative with declared N/s floor |
| Onset time | One-sample backdating; absent/unrecorded onset is null, never manufactured index zero |
| Contact threshold | Fixed N, near-zero fraction after system-weight drop, or mean+k*SD from a declared unloaded window with an explicit floor |
| Contact signal | Raw calibrated force by default; filtered force is an explicit alternative |
| Numerical integration | Trapezoid or cumulative Simpson; declared start and initial velocity |

Zero-lag filtering can anticipate a transition and smooth discontinuities. Raw remains
the impulse default. Derivative onset always filters first. With external weighing and
no recorded pre-movement derivative baseline, derivative onset is explicitly unavailable;
record-start reconstruction can still use the independently supplied initial state.

Reusable outputs include force, acceleration, velocity, displacement, contact state,
paired takeoff/landing events, net impulse over the declared integration interval,
whole-record peak force and separate GRF/net power. Takeoff-velocity height requires
upward velocity. Flight-time height is separately labelled and assumes equal COM
height at takeoff/landing; a truncated flight has no fabricated landing/flight height.
Propulsive/braking phases, DJ strategy, protocol-specific RSI/RFD and interpretations
belong to assessment engines, not an implicit force-platform phase classifier.

## QC, uncertainty and provenance

The existing 25-rule QC catalogue is evaluated on calibrated aligned force. Own weighing
stability, declared ADC/force rails, synchronization, independent assignment and isolated-load
RMS cross-talk checks supplement it. Bilateral correlation does not prove cross-talk.
Missing isolated-load or assignment references remain explicit unavailable checks.
Known instrument identity mismatches reject isolated-load references.

The existing QC-S03 flat-maximum heuristic flags ideal analytical plateaus. Numerical
verification retains those diagnostic rejections instead of distorting the reference
or disabling QC for simulations. Noisy synthetic CSV controls demonstrate the complete
acquisition-to-accepted/flagged-report flow. Real hardware saturation requires its
recorded rails and calibration, not agreement with a shape fixture.

Custom QC files must retain known evaluator IDs and parameter names, with finite valid
parameter domains. The complete resolved rule version, parameters, severities and
canonical digest bind the result, so a changed verdict cannot hide changed QC choices.
The rule catalogue is packaged in the wheel; CI executes the wheel's processor from
outside the source repository to verify that processing does not depend on its cwd.

Uncertainty outputs include weighing SD/standard error (with the independent-sample
assumption), sample/event timing resolution and declared synchronization uncertainty.
For **raw** processing, supplied total calibrated-force and BW error bounds produce
conservative fixed-grid velocity/displacement input-error envelopes. They exclude
quadrature/filter/interpolation error and do not constitute complete empirical
measurement uncertainty. Filtered envelopes are unavailable without filter-operator
propagation; a raw input bound is never falsely presented as bounding a filtered
output. Missing calibration error bounds remain null. The conditional method/rate
sensitivity tables are numerical differences, not confidence intervals.

Every result binds the full method settings, applied plate mappings/zeroing, original
calibration/zeroing/acquisition records, subject/session/test context, input digests,
source file hashes/provenance, weighing/isolated-load references, resolved QC and library
versions. Vendor-reported values do not enter reconstruction or expected-result tests.

## Verification and remaining qualification

[Generated numerical evidence](force-platform-verification.md) and its
[252-row sensitivity table](force-platform-verification.csv) cover single/dual plates,
CMJ/SJ/DJ/repeated hops and all three requested rates. Exhaustive filter/integration/onset
cross-products and separate contact/start variants are tested against independent
RES-669 mathematics. Calibration, geometry, timebase, record identity, covariance
limitations, no-onset, malformed QC/configuration and file processing have independent
positive/negative controls. Refresh the numerical evidence with:

```sh
uv run python tests/test_force_platform_verification.py --write
```

This delivers a working engineering engine. No device is promoted to
ENGINEERING_VERIFIED or SCIENTIFICALLY_QUALIFIED by this PR. Actual instrument calibration,
compliance, cross-talk, placement, clock and empirical uncertainty evidence are still
required; G/H assessment qualification and L agreement studies belong to RES-671/672
and RES-703. All eight families and 15 subfamilies remain in the ledger. RES-670 adds
actual code/test evidence, records partial E/F/I/M progress, and replaces obsolete "engine absent" blockers with the
remaining measured-device and assessment blockers. No paid model evaluation or
DynamisSimBench change is involved.

The repeated-hop evidence uses the explicit five-hop shortened analytical fixture, not
a claim of qualification for a complete standardized 10/5 assessment.
