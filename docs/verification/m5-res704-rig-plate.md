# RES-704 synthetic rig and plate comparison

This scope is the frozen `rig-plate-net-peak-pair@1` manifest and its four
original checkpoint and artifact bindings. The paired protocol uses one ideal
fixed-bar IMTP, direct calibrated-in-newtons synthetic arrays for one rig and
one single plate, one exact native 1,000 Hz timebase and a prescribed coaxial
quasistatic vertical load-transfer model.

All fourteen A–M and O roles apply. Each scoped receipt is bound to **both**
`load_cell_isometric.isometric_rig` and `force_platform.single`.

| Role | Exact paired dependency |
| --- | --- |
| A | Separate rig force and single-plate force acquisition paths and native source digests |
| B | Direct SI passthrough, distinct prescribed tare values and a declared matched +z load path |
| C | Exact rig and plate native grids, shared counter identity and epoch, and common subject, session, trial and protocol |
| D | Finite single-arm inputs and separate complete baseline and RFD support before an eligible comparison |
| E | Unfiltered baseline-corrected force and endpoint RFD under the same named convention on each trace |
| F | Exact reconstruction of each arm's net peak and 100 ms RFD |
| G | Separate strict positive onset searches from the last baseline sample with one-sample backdating, and two complete windows |
| H | Signed rig peak minus plate peak of matched vertical forces, with the nominal declared absolute tolerance |
| I | Worst-case sum of the two declared deterministic absolute net-peak bounds |
| J | Independent closed forms, asymmetric witnesses, separate analytical key author and scientific reviewer |
| K | Refusal of axis, frame, transfer, identity, epoch, rate, baseline bracket and window faults |
| L | Single-pair synthetic discrepancy, signed order and no inferences about variance, reliability or interchangeability |
| M | Both internal sample digests, immutable artifact/revision identity, released input provenance, sealed references and reproducibility |
| O | Synthetic analytical scope and DEV-006 device-specific interpretation boundary |

G applies because both RFD operands require an independently resolved onset.
The rig uses the production load-cell acquisition's direct-force path. The
plate uses the production single-platform acquisition with its declared +z
frame and its own prescribed pre-effort baseline window. The task-level event kernel uses
the frozen signed rule for both acquired signals. The wider platform
movement-onset routines use different search and force conventions.
The first above-threshold sample at the last baseline sample has no supported
bracket under this reference, and the estimator requires the complete 100 ms
native interval. Supplemental nonlinear and asymmetric traces vary the baseline,
onset, RFD and peak time in each arm. Their peaks occur at different samples.

The statistic is a **descriptive single-pair difference**, rig minus plate.
Equality of two computed net peaks describes their mathematical summaries.
Their comparison under the stipulated transfer model describes analytical
agreement only. The 8 N threshold classifies the nominal discrepancy. The
2 N and 3 N peak bounds describe conditional worst-case measurement
perturbations, using the triangle inequality without distribution or
independence assumptions. They supply no between-trial variation,
interdevice reliability, or Bland–Altman limits. RES-674 agreement and
reliability require multiple pairs, so neither is evaluated in this task.
Real compliance, calibration uncertainty, line-of-action validity and
instrument distributions remain unassessed. DEV-006 concerns a specific
measured device and setting. This synthetic exercise establishes no hardware
interchangeability.

The prespecified absolute tolerance is 1e-8 for quantity, radius and named
numeric evidence, with zero relative tolerance. Private operands are
canonically rounded to ten decimals. Methods, units, estimands, citations,
trial identity, refusal reasons, evidence identities and claims must match.
Require four independently authored oracle decisions, zero false passes or
failures on controls, unchanged legacy results, verified shipped code and
source hashes, isolation, deterministic replay and complete workspace cleanup.
