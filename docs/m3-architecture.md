# M3 architecture: quality parity across device families

Issue RES-692. Machine-readable source of truth: `qualification/ledger.toml`.
Generated status: [qualification-ledger.md](qualification-ledger.md). Validate with
`uv run dbs qualification validate`.

## Correction

The first M3 plan built a reference engine around force-platform jump curves. Other
instruments were treated as secondary. That plan is withdrawn. **Every supported
measurement-device family is developed to the same scientific and engineering
qualification standard.** This is *quality parity*: every family meets the same
dimensions with the same rigour. It does not mean every family uses the same
algorithms.

## Three engines

```
 raw device data ──► DEVICE PROCESSING ENGINE ──► observations ──► ASSESSMENT SCIENCE ENGINE ──► protocol quantities
 (per family,         calibration, units, axes,    (calibrated,      onset, phases, reps,           (estimands with
  RES-670,             timing, integrity,           timestamped,      metric definitions             method_spec and
  RES-695..701)        filtering or event logic,    with provenance)  and variants                   uncertainty)
                       reconstruction               RES-671 jumps, RES-672 isometric, RES-673 sprint/COD
                                                              │
                                                              ▼
                                         AI BENCHMARK ENGINE (M4/M5)
                                         tasks, graders and harness that evaluate an AI
                                         system doing this scientific work
```

| Engine | Question it answers | Owns dimensions | Issues |
| -- | -- | -- | -- |
| Device processing | How does the instrument produce observations? | A–F, I–M, O (per family) | RES-670, RES-695 to RES-701 |
| Assessment science | How does a defined protocol produce scientifically meaningful quantities? | G, H | RES-671, RES-672, RES-673 |
| AI benchmark | How is an AI system evaluated on processing and interpreting them? | N | RES-676 to RES-682, RES-704 |

Shared infrastructure serves all three engines:

* The cross-family analytical substrate (RES-669) supplies ground-truth motion models,
  per-family observation models and artifact injectors.
* The statistics library (RES-674) supplies reliability, agreement and individual change.
* Cross-device agreement studies (RES-703) cover dimension L.

## Device families in scope

| # | Family | Paradigm | Device engine |
| -- | -- | -- | -- |
| 1 | Force platforms and dual force plates | continuous signal | RES-670 |
| 2 | Linear position transducers and encoders | continuous signal | RES-695 |
| 3 | Load cells and isometric dynamometers | continuous signal | RES-696 |
| 4 | Nordic hamstring and hip-strength devices | continuous signal | RES-697 (on RES-696) |
| 5 | Timing gates and contact systems | **discrete event** | RES-698 |
| 6 | Radar and laser velocity systems | continuous signal | RES-699 |
| 7 | GNSS and local-positioning systems | position track | RES-700 |
| 8 | Inertial measurement units | inertial | RES-701 |
| 9 | Additional classes | declared in contract | admission contract, RES-702 |

## Qualification dimensions

Every family must meet all fifteen dimensions. What each one *means* depends on the
paradigm, and that difference is the point of parity without identical algorithms:

| Id | Dimension | Continuous signal | Discrete event | Position track | Inertial |
| -- | -- | -- | -- | -- | -- |
| A | Acquisition and raw-data contracts | sampled channels | event lists | positions, Doppler speed | specific force, angular rate |
| B | Calibration, units, axes, geometry | gain, offset, lever arms, cable geometry | beam height and type, distances | datum, local frame, anchors | bias, scale, sensor-to-segment frame |
| C | Sampling, timestamps, synchronization | rate, anti-aliasing | clock resolution, trigger latency | native vs interpolated rate | rate, clock drift, sync events |
| D | Signal or event integrity | offset, drift, clipping, dropouts | missing, duplicate, out-of-order events | HDOP, satellites, jumps | saturation, slip |
| E | Filtering or event processing | low-pass, integration variants | debounce, beam voting, pairing; **no continuous filters** | smoothing, projection | orientation, gravity removal |
| F | Device-specific reconstruction | velocity, displacement, torque | splits, flight and contact times | distance, speed, acceleration | free acceleration, flight time |
| G | Assessment-specific event detection | onset, phases, reps | protocol phases from events | efforts, sprints | take-off, landing, steps |
| H | Metric definitions and variants | named estimands with `method_spec` | same | same | same |
| I | Uncertainty and sensitivity | per method, rate and geometry | trigger and latency error | rate, smoothing, dwell | placement, orientation |
| J | Independent references | closed-form truth, independent implementations, empirical criteria | same | same | same |
| K | Fixtures | positive, negative and adversarial, per paradigm | same | same | same |
| L | Cross-device agreement | studied, never assumed (RES-703) | same | same | same |
| M | Provenance and reproducibility | hash-chained bundles, seeded generators | same | same | same |
| N | AI task families with qualified grading | M4/M5 task sets with grader controls | same | same | same |
| O | Explicit limitations | per family, sourced | same | same | same |

## Qualification states (computed, never declared)

| State | Condition |
| -- | -- |
| `NOT_STARTED` | Every dimension absent |
| `IN_PROGRESS` | Anything between |
| `SPECIFIED` | Every dimension at least specified |
| `ENGINEERING_VERIFIED` (**M3 gate**) | Computational-reference readiness (Policy A): A–M and O each verified by a valid independent receipt, an independent reference, no open computational blocker; N excluded. Says nothing about physical-device validity |
| `SCIENTIFICALLY_QUALIFIED` | Every dimension verified, measured empirical records for every M3 dimension, qualified grading (N), plus independent review (M8, RES-688) |

A family is reported as **blocked** whenever a dimension carries a *scientific*
blocker, such as a missing source, missing data or an unresolved scientific question.
*Dependency* blockers are engineering work owned by another issue.

## Rules the validator enforces

1. **Computed state.** A `declared_state` that disagrees with the dimensions fails.
2. **No qualification by import.** Adapter imports and vendor reproduction may support acquisition (A) only. A family is never qualified because its adapter
   reads a CSV, or because the software reproduces vendor-reported numbers.
3. **No self-validation.** A reference whose `implementation` is the engine it
   `verifies` is rejected, and vendor output is never a reference. RES-669 also adds
   an import-boundary test, so generators cannot import engines and engines cannot
   import generator internals.
4. **Paradigm-appropriate processing.** Continuous-signal filters are rejected for
   discrete-event families.
5. **No assumed interchangeability.** Each declared non-equivalence needs a study. An
   `interchangeable` or `convertible` verdict needs evidence and an analytical or
   empirical basis and accepted review for interchangeability. Analytical agreement alone never establishes empirical interchangeability. Declared examples include:
   * GNSS speed vs radar velocity.
   * LPT cable vs vertical displacement.
   * LPT vs force-plate power.
   * Contact-mat or IMU flight time vs force-plate flight time.
   * IMTP on a load-cell rig vs on a plate.
   * Gate splits vs radar-derived splits.
   * Vendor-corrected vs raw Nordic force.
6. **Vendor definitions differ.** Commercial force, torque, velocity, power and RFD
   definitions are never assumed interchangeable between vendors (QP-06, DB-07).
7. **No silent narrowing.** Every in-scope family must feed an assessment engine, and
   none can be removed without a visible ledger change.

## Issue graph

```
RES-692 contract + ledger
  ├─► RES-693 schema v2 ──► RES-669 analytical substrate ──┐
  ├─► RES-694 registry extension ─────────────────────────┤
  │                                                       ▼
  │   device engines: RES-670 force plate · RES-695 LPT · RES-696 load cell ─► RES-697 Nordic/hip
  │                   RES-698 timing/contact · RES-699 radar/laser · RES-700 GNSS/LPS · RES-701 IMU
  │                                                       │
  │   assessment engines: RES-671 jumps · RES-672 isometric · RES-673 sprint/COD
  │   RES-674 statistics ─► RES-703 cross-device agreement
  │                                                       ▼
  └──────────────────────────────────────────────► RES-675 M3 gate (all families)
RES-702 admission contract (additional classes)
M4/M5: RES-679 grader qualification ─► RES-680..682, RES-704 (dimension N)
M8: RES-688 independent review ─► SCIENTIFICALLY_QUALIFIED
```

## Current status (2026-11-14)

No family has reached the M3 gate. All eight are `IN_PROGRESS`. Seven carry
scientific blockers, and each is reported, not hidden:

* **LPT/encoders, load cells, GNSS/LPS and IMU:** no device-validity or
  measurement-error claims in the registry (RES-694).
* **GNSS/LPS and IMU:** no raw-data schema (RES-693) and no audited raw dataset for
  empirical agreement.
* **Timing/contact and radar/laser:** device-level error sources are not yet sourced.
* **Nordic/hip:** the isometric-hamstring registry gap blocks part of H.
* **Force platforms:** no scientific blocker, but no engine work exists yet.

Families that cannot reach the gate will be reported as blocked in the RES-675 report.
M3 is not complete until every in-scope family passes, or until the owner formally
changes the approved scope.

## Constraints

At least two meaningful atomic commits per issue. Independent verification. Explicit
scientific assumptions. Qualified data provenance. No paid model evaluations. No
modifications to DynamisSimBench.

## RES-692 reconciliation

PR #4 merges the original commits `2654c33` and `856ece4` without rewriting either
history. The stage-aware subfamily ledger supersedes the family-only model. Its
validator carries forward evidence resolution, declared paradigm restrictions,
reference independence, unresolved agreement checks and blocker-aware gates.
The original three-engine architecture, A–O definitions and issue graph are retained.
The original claim that N must be specified at M3 is corrected: N belongs to M5.
Evidence remains provisional and no subfamily is verified.

## Assessment routing retained from the original contract

- Jumps: CMJ, SJ, single-leg CMJ, drop jump, 10/5 (RES-671): cmj, sj, slcmj, dj, rj_10_5; devices: force_platform, lpt_encoder, timing_contact, imu.
- Isometric and dynamometry: IMTP, Nordic, hip squeeze, isometric hamstring (RES-672): imtp, nhe, hip_add_abd, iso_ham; devices: force_platform, load_cell_isometric, nordic_hip.
- Sprint, sprint F-V profile and change of direction (RES-673): sprint_gates, sprint_fv, cod_505, ift_30_15, rsa; devices: timing_contact, radar_laser, gnss_lps.

## Admission and principles

A new device class enters only through a versioned scientific evidence contract covering dimensions A-O. It starts with scope 'proposed', has no task cells and no grading, and cannot exceed SPECIFIED until it is approved into scope.

- QP-01: Every in-scope device family is held to the same qualification dimensions A-O. Parity is in qualification quality, not in identical algorithms.
- QP-02: Family qualification states are computed from dimension evidence, never declared. A hand-set label that disagrees with the computed state fails.
- QP-03: Adapter imports and vendor reproduction support acquisition A only; neither establishes independent verification.
- QP-04: No reference may validate itself. A reference produced by the same implementation as the engine it verifies is rejected; vendor output is never a reference (DB-07).
- QP-05: Processing methods must suit the family's paradigm. Continuous-signal filtering is not applied to discrete timing or contact events.
- QP-06: Measurements from different devices are different estimands until an agreement study rules otherwise. Examples: GNSS-derived speed is not a radar velocity trace; LPT cable displacement is not vertical displacement without geometry; commercial force, torque, velocity, power and RFD definitions are not interchangeable between vendors.
- QP-07: Three engines are kept separate: the device processing engine (how an instrument produces observations), the assessment science engine (how a protocol produces scientific quantities) and the AI benchmark engine (how an AI system is evaluated on that work).
- QP-08: M3 is complete only when every in-scope subfamily computes ENGINEERING_VERIFIED, which under the owner-approved Policy A means computational-reference readiness (see "M3 contract (Policy A)"). Families that cannot are reported as blocked, never silently removed from scope. SCIENTIFICALLY_QUALIFIED needs independent review (M8) and qualified grading (N).
- QP-09: Project constraints: at least two atomic commits per issue, independent verification, explicit scientific assumptions, qualified data provenance, no paid model evaluations, and no modifications to DynamisSimBench.

## Declared estimand non-equivalences

- NE-001: gnss_lps:speed.gnss vs radar_laser:velocity.radar: GNSS-derived speed is not equivalent to a radar velocity trace: rates, smoothing, latency and the body point all differ. Study: RES-703; verdict unknown.
- NE-002: lpt_encoder:displacement.cable vs force_platform:displacement.com_vertical: LPT cable displacement is not vertical centre-of-mass or barbell displacement unless the cable geometry is declared and the attachment point's motion relative to the body is accounted for. Study: RES-703; verdict unknown.
- NE-003: lpt_encoder:power.lpt vs force_platform:power.force_plate: LPT power (mass x acceleration x velocity at the attachment point) and force-plate power (GRF x COM velocity) are different estimands. Study: RES-703; verdict unknown.
- NE-004: timing_contact:flight_time.contact_mat vs force_platform:flight_time.force_plate: Contact-mat flight time includes switching latency and threshold differences, so it is not force-plate flight time without an agreement study. Study: RES-703; verdict unknown.
- NE-005: imu:flight_time.imu vs force_platform:flight_time.force_plate: IMU flight time depends on mounting site and event thresholds, so it is not force-plate flight time without an agreement study. Study: RES-703; verdict unknown.
- NE-006: load_cell_isometric:peak_force.imtp_rig vs force_platform:peak_force.imtp_plate: IMTP peak force on a load-cell rig and on a force plate differ in rig compliance, body-weight handling and onset detection. Study: RES-703; verdict unknown.
- NE-007: timing_contact:split_time.gates vs radar_laser:split_time.derived: Gate splits are triggered by the first body part to break the beam; radar- or laser-derived splits track a body point. They are not interchangeable. Study: RES-703; verdict unknown.
- NE-008: radar_laser:velocity.radar_doppler vs radar_laser:velocity.laser_differentiated: Radar Doppler velocity and velocity differentiated from laser distance have different noise models and processing. Study: RES-703; verdict unknown.
- NE-009: gnss_lps:speed.doppler vs gnss_lps:speed.positional: Device Doppler speed and speed differentiated from positions are separate estimands, and both depend on vendor filtering. Study: RES-703; verdict unknown.
- NE-010: nordic_hip:force.vendor_corrected vs nordic_hip:force.raw_cell: Vendor-corrected Nordic or squeeze force is not raw cell force; commercial force and torque definitions are not interchangeable between vendors. Study: RES-703; verdict unknown.

## Processing catalogue

- `raw_passthrough` (continuous_signal, position_track, inertial): No filtering; the documented default for impulse-based force metrics.
- `butterworth_zero_lag` (continuous_signal, position_track, inertial): Low-pass Butterworth, forward-backward, order and cutoff declared.
- `chebyshev_zero_lag` (continuous_signal): Low-pass Chebyshev type I, forward-backward.
- `moving_average` (continuous_signal, position_track, inertial): Centred moving average, window declared.
- `trapezoid_integration` (continuous_signal, inertial): Trapezoidal numerical integration.
- `simpson_integration` (continuous_signal, inertial): Composite Simpson integration.
- `finite_difference` (continuous_signal, position_track): Central finite-difference differentiation.
- `spline_differentiation` (continuous_signal, position_track): Smoothing-spline fit, then analytic derivative.
- `cosine_correction` (continuous_signal): Correct line-of-sight radar or laser measurements for mounting geometry.
- `event_debounce_refractory` (discrete_event): Suppress repeated triggers inside a declared refractory window.
- `multi_beam_voting` (discrete_event): Accept a gate event only when the declared beams agree.
- `contact_pairing` (discrete_event): Pair contact on/off events into contact and flight phases.
- `trigger_latency_correction` (discrete_event): Apply a declared start-trigger or switching-latency offset.
- `position_projection_local` (position_track): Project WGS-84 positions to a declared local tangent-plane or pitch frame.
- `doppler_speed_passthrough` (position_track): Keep device-reported Doppler speed as its own estimand.
- `orientation_complementary` (inertial): Complementary-filter orientation estimate.
- `orientation_kalman` (inertial): Kalman-filter orientation estimate.
- `gravity_removal` (inertial): Remove gravity from specific force using the estimated orientation.

## M3 contract (Policy A, owner-approved)

**Approved.** M3 verifies the *computational reference*: given declared inputs and conditions,
the software computes the specified quantity correctly, as shown by independent checks. It is
*verification* in the NIST/ASME sense (are we solving the equations right) and says nothing
about *validation* (does a physical instrument measure the right thing). Physical-device
validity, calibration and metrological traceability (JCGM 200:2012, VIM) need measured data,
a criterion and an uncertainty statement; they are recorded separately in
`qualification/empirical.toml` and are never inferred from synthetic or analytical evidence.

One canonical evaluator (`qualification/gate.py`) decides M3 for `ledger.gate`, `state.m3_gate`
and the `dbs qualification gate|states` commands. A dimension of a subfamily is verified only
when its rating is `verified` **and** at least one valid receipt covers it. A receipt
(`qualification/receipts.toml`) is not a file path or an evidence-kind string. It names:

* the implementation symbols under test (under `src/`);
* an independent reference: a closed form, an independent implementation or a published
  worked example, in a file that is not an implementation file and does not call
  engine/assessment code;
* the tests that compare them (real `test_*` functions with assertions, not skipped);
* a positive numeric tolerance with unit and basis, a substantive claim and independence
  statement; and
* SHA-256 digests of the syntax tree of every cited symbol, so any edit invalidates the receipt
  until it is re-issued and independently re-reviewed.

`tests/test_receipts_execute.py` additionally runs every cited test, so no receipt can rest on
a failing test.

### Computational reading of A–M and O

| Dim | M3 (computational) requirement | What stays empirical |
| -- | -- | -- |
| A | executable admission contract: native/vendor stage, units and integrity refusals tested | audited native raw-data acquisition from the device |
| B | declared calibration and geometry records are typed, applied by processing, and missing or contradictory records are refused | measured calibration, axes and geometry records |
| C | clock, latency and synchronisation models are declared, applied and propagated; inconsistent timing is refused | measured clock, latency and synchronisation characterisation |
| D | integrity and QC checks reject injected faults and pass clean controls | study with real fault occurrences |
| E | every processing method variant is compared with an independent reference with stated sensitivity bounds | method comparison on measured data |
| F | reconstructed quantities agree with an independent analytical truth within tolerance | reconstruction checked against a criterion measure |
| G | protocol events are recovered against analytically known event times, with eligibility refusals | detection checked against adjudicated events |
| H | metric definitions are bound to sourced registry claims and executable exactly | domain-expert definition review |
| I | declared uncertainty propagation is checked against an independent analytical result | uncertainty budget from device metrology |
| J | an independent reference implementation or closed form that shares no code with the engine | criterion instrument for the physical quantity |
| K | synthetic fault fixtures cover the declared fault classes with expected verdicts | fixtures of measured fault classes |
| L | analytical cross-device studies are executed and classified, never read as interchangeability | paired measured agreement study (at least 20 independent subjects) |
| M | seeded, hash-bound provenance chain for every processing stage, reproduced exactly | provenance of measured data to the acquisition session |
| O | limitation statements are sourced and emitted with every result | domain-expert review of limitations |

### Analytical versus empirical interchangeability

Dimension L at M3 means the analytical cross-device studies were executed and classified
(`ENGINE_TO_ENGINE`, `ENGINE_TO_INDEPENDENT_CLOSED_FORM`, `CLOSED_FORM_DEMONSTRATION`,
`DIAGNOSTIC_ONLY`, `UNSUPPORTED`). Each comparison's empirical verdict stays `unknown` until
qualified measured paired data exist; an `interchangeable` verdict still needs empirical
evidence and accepted review. Analytical agreement never promotes it.

### Reports

`dbs qualification states` and `docs/verification/engine-v0.1.md` list **computational
blockers** (unverified dimensions and open computational issues, with owners) separately
from **empirical requirements** (what only physical measurement can close). The M4/M5/M8
boundaries are unchanged: N still belongs to M4/M5, and independent scientific review to M8.

### Original RES-675 acceptance, reinterpreted

The original acceptance ("full verification matrix ... cross-implementation agreement ...
report in `docs/verification/engine-v0.1.md` ... open discrepancies with a disposition")
is read as computational verification. Phrases implying device validation (audited native
data, metrology, paired measured agreement) are empirical-track requirements and are not
M3 pass conditions. All eight families and 15 subfamilies remain in scope; none is removed.
