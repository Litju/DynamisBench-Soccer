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
| `ENGINEERING_VERIFIED` (**M3 gate**) | A–M and O verified, N at least specified, at least one independent reference |
| `SCIENTIFICALLY_QUALIFIED` | Every dimension verified, plus a non-analytical independent reference (independent implementation, empirical criterion or published worked example), plus independent review (M8, RES-688) |

A family is reported as **blocked** whenever a dimension carries a *scientific*
blocker, such as a missing source, missing data or an unresolved scientific question.
*Dependency* blockers are engineering work owned by another issue.

## Rules the validator enforces

1. **Computed state.** A `declared_state` that disagrees with the dimensions fails.
2. **No qualification by import.** Evidence of kind `adapter_import` or
   `vendor_reproduction` never counts. A family is never qualified because its adapter
   reads a CSV, or because the software reproduces vendor-reported numbers.
3. **No self-validation.** A reference whose `implementation` is the engine it
   `verifies` is rejected, and vendor output is never a reference. RES-669 also adds
   an import-boundary test, so generators cannot import engines and engines cannot
   import generator internals.
4. **Paradigm-appropriate processing.** Continuous-signal filters are rejected for
   discrete-event families.
5. **No assumed interchangeability.** Each declared non-equivalence needs a study. An
   `interchangeable` or `convertible` verdict needs evidence and an analytical or
   empirical basis. Declared examples include:
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
