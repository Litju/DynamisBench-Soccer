# RES-704 load-cell torque reference

This campaign targets the existing eight checkpoints of `load-cell-net-torque-ramp@1`.
The task, checkpoint order, source bytes and 1e-8 absolute grading tolerance are frozen.
The supported computation is unfiltered peak external knee torque above the unloaded
baseline, in N·m, for a fixed right distal-shank strap.

| Receipt role | Dependency for this task | Evidence required before admission |
| --- | --- | --- |
| A | One native synthetic voltage channel with declared sensor axis and device class | Canonical bundle and source validation before native conversion |
| B | Readout V→N or inverse sensitivity N→V, declared gain and offset, perpendicular lever | Calibrated trajectory against the designed ramp and calibration/attachment refusal |
| C | Native 1000 Hz on one counter, zero relative clock correction and exact grid | Native reconstructed grid and clock provenance, with inconsistent-rate controls |
| D | Voltage rail integrity and valid pre-effort baseline | Closed-rail fault and unloaded voltage with nonzero calibrated tare |
| E | Both force streams unfiltered, 100 ms endpoint difference and trapezoidal impulse | Unfiltered trajectory, timing and window checks against independent values |
| F | Baseline and signed force onset | Strict threshold and one-sample backdate, with no-rise and short-record refusals |
| G | Right-side fixed-knee protocol, declared eligible knee angle and known distal-shank attachment | Reject out-of-range angle, wrong limb and wrong or unknown attachment |
| H | Gross and net force are separate from external knee torque | Signed force ramp and exact lever scaling, no gravity correction |
| I | Conditional deterministic rectangular net-force and lever errors | Evaluate the four corner products, including the force-by-lever error term |
| J | Independent numerical truth | Closed-form ramp, separately authored checkpoint answers and reviewer re-derivation |
| K | Fault injection | Replay all six source faults and additional boundary faults through the task procedure |
| M | Traceable metadata and immutable artifacts | Exact fixture digests, clock identity, scope, private-answer and review bindings |
| O | Interpretation and evidence boundary | External joint torque, synthetic evidence, and the specific IMTP context of DEV-006 |

L does not enter this single-cell estimate. There is no cross-device comparison.
The task-level verification of C, E, H, I and K cannot qualify the broader load-cell
engine for all timing, filtering, torque, uncertainty and fault variants. These roles
require task-scoped receipts. Full M3 remains under RES-675 and RES-696 retains its
original acceptance history.

The first 101 samples are unloaded in the synthetic rig model, including the initial
electrical tare. The ramp starts later. The exact reference slice has a resolved
pre-threshold bracket before onset and all 100 ms of post-onset support. An effort
already above threshold at the start of the search lacks a resolved bracket in this
engine and is outside that computed slice. No analog anti-aliasing or real-device
offset stability is inferred from the synthetic array.

The acceptance threshold is absolute error at most `1e-8` for quantity, radius and
ten-decimal numerical evidence. Units, estimand, method, trial and evidence identity,
refusal reason, citations and allowed claims must match exactly. Every oracle
checkpoint must pass. Prespecified controls must have zero false accepts and zero false
rejects. Record the actual scores of all weaker baselines. Trials must replay with
identical decisions and hashes, without execution, leakage or cleanup failures.

The uncertainty radius is conditional on the supplied bounds on net force and lever.
It does not estimate measurement coverage probability or include uncertainty from
observed clock error, rig compliance, protocol execution or hardware validity.
The perpendicular distance runs from the joint centre to the cell's line of action.
The result quantifies external knee torque about that centre. Muscle forces, internal
joint torque and interchangeability of load-cell and force-platform IMTP are outside
this evidence. [DEV-006 source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346244/)
