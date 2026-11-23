# M3 disposition under Policy A (RES-675)

Computed from `qualification/ledger.toml`, `receipts.toml` and `empirical.toml` by the canonical evaluator (`dbs qualification states`). Policy A: M3 = computational-reference readiness. Physical-device validity is reported separately and is not an M3 pass condition.

## Result

**M3 FAIL: 0 of 15 subfamilies are `ENGINEERING_VERIFIED`.** Both `dbs qualification gate --stage M3` and `dbs qualification states` exit 1. The RES-675 gate remains open and blocks RES-676.

98 of 210 A–M/O cells have supported and content-bound receipts. The 112 open cells below still block computational readiness.

## 1. Computational-reference status (15 subfamilies)

| Subfamily | State | Verified (computational) | Open |
| -- | -- | -- | -- |
| force_platform.dual | SPECIFIED | C, D, F, I, J, M | A, B, E, G, H, K, L, O |
| force_platform.single | SPECIFIED | F, G, I, J, L, M | A, B, C, D, E, H, K, O |
| gnss_lps.gnss | SPECIFIED | A, C, D, F, G, J, K, L, M, O | B, E, H, I |
| gnss_lps.lps | SPECIFIED | A, B, C, D, E, F, G, J, K, M, O | H, I, L |
| imu.inertial | SPECIFIED | B, D, E, F, I, J, K, O | A, C, G, H, L, M |
| load_cell_isometric.isometric_rig | SPECIFIED | A, M, O | B, C, D, E, F, G, H, I, J, K, L |
| load_cell_isometric.load_cell | SPECIFIED | A, B, D, F, G, J, L, M, O | C, E, H, I, K |
| lpt_encoder.encoder | SPECIFIED | B, M | A, C, D, E, F, G, H, I, J, K, L, O |
| lpt_encoder.lpt | SPECIFIED | C, F, I, J, M | A, B, D, E, G, H, K, L, O |
| nordic_hip.hip | SPECIFIED | B, H, J | A, C, D, E, F, G, I, K, L, M, O |
| nordic_hip.nordic | SPECIFIED | F, G, H, J, L | A, B, C, D, E, I, K, M, O |
| radar_laser.laser | SPECIFIED | C, F, L | A, B, D, E, G, H, I, J, K, M, O |
| radar_laser.radar | SPECIFIED | A, C, D, F, G, H, I, J | B, E, K, L, M, O |
| timing_contact.contact | SPECIFIED | A, D, E, F, H, I, J, K, L, M | B, C, G, O |
| timing_contact.gates | SPECIFIED | A, D, E, F, H, I, K, L, M | B, C, G, J, O |

Verified cells: 98 of 210.

## 2. Empirical device-validity status (15 subfamilies)

`qualification/empirical.toml` holds **no record**. Every subfamily therefore has no measured evidence for any dimension. Synthetic, analytical, simulated and vendor-reported data are rejected by the validator and can never fill it.

| Subfamily | Measured records | Interchangeability | Known empirical gaps |
| -- | -- | -- | -- |
| force_platform.dual | 0 | unknown | 2 recorded (see ledger) |
| force_platform.single | 0 | unknown | 2 recorded (see ledger) |
| gnss_lps.gnss | 0 | unknown | 2 recorded (see ledger) |
| gnss_lps.lps | 0 | unknown | 2 recorded (see ledger) |
| imu.inertial | 0 | unknown | 2 recorded (see ledger) |
| load_cell_isometric.isometric_rig | 0 | unknown | 2 recorded (see ledger) |
| load_cell_isometric.load_cell | 0 | unknown | 2 recorded (see ledger) |
| lpt_encoder.encoder | 0 | unknown | 2 recorded (see ledger) |
| lpt_encoder.lpt | 0 | unknown | 2 recorded (see ledger) |
| nordic_hip.hip | 0 | unknown | 2 recorded (see ledger) |
| nordic_hip.nordic | 0 | unknown | 2 recorded (see ledger) |
| radar_laser.laser | 0 | unknown | 2 recorded (see ledger) |
| radar_laser.radar | 0 | unknown | 2 recorded (see ledger) |
| timing_contact.contact | 0 | unknown | 2 recorded (see ledger) |
| timing_contact.gates | 0 | unknown | 2 recorded (see ledger) |

The previous per-subfamily `blockers` text is preserved verbatim as `empirical_gaps` in the ledger; `blockers` now holds computational blockers only (none are declared beyond the unverified cells).

## 3. Open computational work, from the independent audit

Prior drafts were PARTIAL, UNSUPPORTED or NOT_AUDITED. The other open cells have no draft receipt. The [evidence gap inventory](m3-evidence-gaps.md) identifies ownership and evidence gaps for the 147 baseline open cells. 35 of those cells now have supported receipts. The [receipt audit](m3-receipt-audit.md) records current verdicts and the evidence still needed.

Recurring gaps (each is concrete engineering, owned by the existing issue for that engine):

* Tests run the *other* subfamily's fixture (single vs dual plate, load cell vs isometric rig, nordic vs hip, LPT vs encoder, GNSS vs LPS, radar vs laser), so the cited evidence does not exercise the subfamily it is filed under.
* Fault-injection coverage asserts injected bytes but not the engine's verdict for most fault classes (Nordic/hip clipping without declared rails, load-cell drift/noise/resampling/dropout, and contact double-contact and latency).
* Tolerance bookkeeping: several receipts record a reference self-test bound instead of the bound the engine tests assert.
* Metric-definition cells (H) lack a test binding each estimand to a sourced registry claim.
* Uncertainty cells (I) mostly assert that sensitivity exists, not that it equals an analytical expectation.
* Provenance (M) and limitations (O) are emitted by only some engines. Radar, laser, IMU, gates and contact still need bound source provenance. The Nordic and hip CLI records source and configuration hashes, and direct processing needs provenance evidence. Force-platform and timing results need emitted limitations. Nordic and hip results need sourced limitation bindings.

## 4. Analytical versus empirical interchangeability

Dimension L is verified only where the analytical study was executed and classified. Every comparison's empirical verdict stays `unknown`; no comparison is `conditionally_interchangeable`.

## 5. Roadmap consequence

M4 stays BLOCKED: RES-676 is blocked by RES-675, which does not pass. The RES-737 development-only prototype is unchanged and unqualified.
