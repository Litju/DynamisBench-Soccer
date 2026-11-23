# RES-704 advanced measurement reasoning

The second M5 slice adds a paired analytical agreement task and five instrument development tasks. Their source observations are synthetic. Admission uses the existing M4 operator catalog, independent review, and revision-bound private answers. The public catalog yields `DEVELOPMENT_ONLY_NOT_QUALIFIED` for these references.

## Paired agreement reference

`paired-lps-radar-agreement@1` uses one position-derived horizontal LPS tag speed per distinct synthetic run. The positions are native 25 Hz field-frame samples on an exact shared measurement clock. The independent radar arm is an ideal field-X projection with a fixed, declared travel-direction cosine. Calculation uses the unfiltered interior three-point derivative and RES-674 Bland–Altman signed bias and descriptive normal-theory limits. The fixed sign convention is LPS minus projection-corrected radar. The task checks complete native acquisition, distinct trial identity and source records, frame, direction, and compatible estimands before comparison.

There are matched, constant-bias, and spread cases. A case with acceptable mean bias and excessive limits checks that both limits enter the tolerance judgment. Other cases withhold agreement for a radial estimand mismatch, failed native QC, duplicated trial identity, or a request to estimate same-method repeatability or individual change from this design. Discrepancies can be attributed to declared synthetic injections in this reference. Measured error causes and real hardware error distributions require separate evidence.

The common-offset uncertainty budget combines independent method offsets after projecting the radial standard uncertainty. The offsets are common across runs within each method and do not shrink with pair count. The k=2 result is conditional on that budget. The descriptive limits are separate from this uncertainty, a confidence interval for bias, and hardware performance. The absolute grading tolerance is `1e-9 m/s` with zero relative tolerance. Derived evidence operands are rounded to 10 decimal places. The independence of the oracle was checked using endpoints, pairwise squared differences for variance, and enumerated common offset outcomes. Production uses the existing native LPS reconstruction and RES-674 agreement implementation.

The comparison binds A–F and H–M and O under the existing cross-device admission kind. G is inapplicable because no protocol event is estimated. Supplemental receipts `RCP-gnss_lps-lps-L-paired-speed` and `RCP-gnss_lps-lps-I-paired-speed` bind the particular audited comparison and uncertainty. Existing LPS receipts supply the other applicable roles. The supplemental evidence leaves the M3 predicates and all ratings intact. The reference covers a native LPS arm and an analytical radar comparator. Native radar calibration, latency, direction recovery and deployment need separate evidence.

Registry claims DEV-005, DEV-008 and ST-004 support interpretation. The numerical answers come from the independent analytical design. [Linke et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6056042/) · [Fornasier-Santos et al.](https://www.mdpi.com/1424-8220/22/22/8610) · [Bland and Altman](https://pubmed.ncbi.nlm.nih.gov/2868172/)

## Development-only instrument references

| Task and source subfamily | Main estimand and controls | Open receipt roles for the proposed scope |
| --- | --- | --- |
| `lpt-vertical-attachment@1`, `lpt_encoder.lpt` | Vertical attachment speed and a conditional translated-load power operand. Exercises cable angle, timing, slack, and unsupported plate or COM power. | A, B, D, E, H, K, O for native reconstruction. |
| `encoder-rollover-attachment@1`, `lpt_encoder.encoder` | Signed vertical attachment displacement, declared scale and direction, modular rollover and endpoint quantization. Refuses absent scale or geometry and ambiguous or excessive count steps. | A, C, D, E, F, H, I, J, K, O for native reconstruction. |
| `load-cell-net-torque-ramp@1`, `load_cell_isometric.load_cell` | Net joint torque from raw voltage, declared calibration, zeroing and moment arm, with onset, RFD and impulse evidence. Tests missing calibration, saturation, protocol and attachment faults. | C, E, H, I, K for native protocol assessment. |
| `rig-plate-net-peak-pair@1`, `load_cell_isometric.isometric_rig` | Signed paired net peak-force discrepancy under an ideal aligned rigid model, with RFD as evidence. Refuses mismatched axis/definition and unresolved timing. | B, C, D, E, F, H, I, J, K and L for the comparison. The broader protocol-specific assessment needs G. |
| `imu-frame-observability@1`, `imu.inertial` | Field-frame sensor-point position under explicit calibration and initial conditions. Separate low-specific-force flight signature and refusals for unobservable states, clock gaps, clipping and COM inference. | A, C, H, M for native reconstruction, plus G for protocol-event assessment. |

These 30 checkpoints use the offline runner and produce diagnostic computations or refusals. Every checkpoint is excluded from scoring. The load-cell model exercises signed calibration and lever uncertainty against an independent synthetic ramp. The rig includes aligned and incompatible definitions and timing. The IMU development case includes a gravity-magnitude ambiguity with two physically compatible states. [Motion development evidence](m5-motion-development.md) · [Force development controls](../../tests/test_m5_force_development.py) · [DEV-006 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346244/)

## Qualification boundary

The applicable dimension N ratings remain absent. A scored exact reference and development coverage do not satisfy the original family-wide N acceptance. Remaining RES-704 scope includes admitted LPT and encoder geometry and power, admitted load-cell and rig calibration, torque and IMTP agreement, admitted IMU orientation, sensor placement and flight-time estimands, and broader valid paired and repeat-session designs. Full M5 and RES-704 remain open. Neither RES-675 nor the full M3 gate is closed by this work.

The measured-device validity and interchangeability questions remain open. Synthetic agreement is limited to the declared model and does not establish device accuracy or general benchmark difficulty. The checked-in diagnostics do not constitute a confidential holdout. Independent human scientific qualification remains M8.
