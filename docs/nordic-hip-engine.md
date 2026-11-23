# Nordic and hip-strength engine (RES-697)

`dbs nordic-hip process <bundle> --config <json> [--out report.json]`

The engine wraps the independently reviewed load-cell core (RES-696: calibration, timing,
rail/capacity checks, Butterworth filtering, onset variants) behind an explicit admission
step. The load-cell engine itself still refuses `nordic_dynamometer` and `hip_dynamometer`
bundles; `from_bundle(..., admitted_classes=...)` is the only door and only this engine
uses it.

## Estimands

* **Strap/cuff force at the device (N).** Not hamstring or hip-muscle force.
* **Torque (N·m)** = strap force × a *declared* lever arm. It needs both `lever_arm_m` and a
  `moment_arm_definition`. Missing values produce a torque refusal. A configured arm
  that differs from the declared bundle arm by more than 1 nm rejects the acquisition.

## Mandatory declarations (refused when absent or contradictory)

| Declaration | Refusal |
| -- | -- |
| `force_basis` (`raw_calibrated` / `vendor_corrected`) | validation error; carried into every output and never pooled |
| side of every channel | duplicate side, `left`+`unilateral` mix, or contradiction with bundle side metadata |
| Nordic needs one left and one right channel | any other layout |
| protocol vs device class | `hip_squeeze` on a Nordic bundle and vice versa |
| knee angle (Nordic) / hip flexion angle (hip squeeze) | protocol geometry unknown |
| measured angle channel (`deg`) | angle, work and break metrics refused; never recovered from force or from an assumed rate |

## Metrics and variants

Per side: peak force, peak time, time to peak (record start and onset), impulse to peak and
over the record, best mean force over a declared window, peak torque, per-repetition peaks
(`single_trial` or `detect` with declared height, prominence and spacing). With a measured
angle: angle at peak force, eccentric work (∫ torque dθ) and break point, with variants
`peak_force_angle` and `angular_acceleration` (angular velocity above a declared multiple of
the controlled rate for a declared duration; refused if no positive controlled rate).
Bilateral: summed peak force and one of two named asymmetry definitions.
Onset variants: `first_exceedance` (no backdating) plus the load-cell `5sd`, `reverse_scan`,
`first_derivative`, `absolute_N`.

For unilateral hip squeeze, registry claim HG-004 (Light and Thorborg) binds the
squeeze-force and torque estimands to declared position and lever geometry.
Peak cuff force is the greatest force sample in the selected effort. Best mean
force is a repository-defined metric using consecutive samples in the declared
window. Mean peak force
averages the detected per-effort maxima. The output withholds bilateral metrics
when only a unilateral channel is available.

## Uncertainty

Conditional sensitivity only: gain ±1 %, low-pass cutoffs, and torque uncertainty propagated
from a declared lever-arm uncertainty. It is not a complete uncertainty budget.

## Verification

`tests/test_nordic_hip.py` compares against closed forms derived from the RES-669 eccentric
ramp (peak time = break angle / rate, impulse = P·t/2, work = ½·P·L·θ), an independent
accelerating-angle fixture for the kinematic break point, negative controls for every
refusal above, injected clipping and dropped-sample corruption, a repetition fixture, the
import boundary and the CLI.

Engineering calculations only. No device, athlete-risk or physiological qualification, and
no raw-versus-vendor-corrected agreement claim (RES-703).
