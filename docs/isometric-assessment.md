# Isometric and dynamometry assessment (RES-672)

`dbs assess isometric <trial bundle>... --config config.json [--out report.json]`

Config: `isometric` (`IsoSpec`), `method` (load-cell engine method), `cell` (load-cell
declarations). One bundle stem per trial. Nordic eccentric trials go through the Python API
`assess_nordic` (it consumes Nordic engine results, RES-697).

## Protocols

`imtp`, `isometric_hamstring`, `hip_adduction`, `hip_abduction` (load-cell core, RES-696)
and `nordic_eccentric` (Nordic engine). Required joint angles come from the bundle geometry
(`knee_angle_deg` for IMTP and hamstring, `hip_flexion_deg` for hip tests). An undeclared
angle, or one outside the declared protocol range, makes the record ineligible.

## Quantities and variants

Per trial: onset (variants `5sd`, `reverse_scan`, `first_derivative`, `absolute_N`), peak and
net peak force, time to peak, RFD and net impulse over declared windows (0–50 … 0–250 ms),
peak torque from the declared lever arm. Policies for repeated trials: `best_peak`,
`mean_peak`, `first_valid`, `all_valid`, with a declared consistency limit on peak spread.
Relative force needs a declared body mass; otherwise it is refused.

A trial is invalid when the device verdict is reject, no onset is found, force starts loaded
beyond a declared baseline limit, or the peak lies at the record edge (the effort may have
been cut off). Invalid trials never enter a summary.

## Uncertainty and sensitivity

Calibration gain perturbations scale force, RFD and impulse; headline RFD and impulse are
recomputed under every onset variant and reported next to the declared one; rig compliance
is recorded as not modelled.

## Verification

`tests/test_isometric_assessment.py` derives RFD(0–w) = (P−b)/R and impulse = (P−b)w²/(2R)
from the RES-669 ramp, checks policies, consistency, geometry eligibility, edge and baseline
invalidation, onset-variant reporting, gain sensitivity, a clipped trial (rejected, never
valid), Nordic policy and force-basis separation, import boundary and the CLI.
Engineering calculations only; no IMTP-on-load-cell versus IMTP-on-plate equivalence claim.
