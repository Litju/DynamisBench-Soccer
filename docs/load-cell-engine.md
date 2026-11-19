# Load-cell and isometric-dynamometer engine (RES-696)

`dbs load-cell process <bundle> [--config config.json] [--sensitivity] [--out report.json]`

Raw voltage, count or force from a uniaxial cell becomes calibrated force, onset, peak, RFD and
impulse windows, and (only with a declared lever arm and moment-arm definition) torque.
Force at the cell and joint torque are different estimands. Gravity correction is a named
variant, never a default.

* Calibration: a declared record or calibration points with a fit and residual check, or an
  explicit kgf conversion; no calibration is ever invented.
* Timing: uniform grid required for filtering and windows; non-uniform records refuse RFD,
  impulse and filtering and say so.
* Integrity checks: rate error, anti-aliasing declaration, calibration fit, rails and capacity
  saturation, temperature drift from declared coefficients, zero offset, creep, unit
  mislabel (declared plausibility bounds), off-axis (not applicable for a uniaxial cell).
* Variants: onset `5sd`, `reverse_scan`, `first_derivative`, `absolute_N`; peak and RFD filters;
  gain, cutoff and onset sensitivity.
* Limitations are part of every report: IMTP on a load cell is not interchangeable with IMTP on
  a force plate (RES-703), rig compliance is recorded and not modelled, a uniaxial cell cannot
  see off-axis loading, unit-mislabel detection is a heuristic, creep and voluntary decline are
  indistinguishable, torque assumes a perpendicular line of action.

Verification: `tests/test_load_cell_engine.py` and `tests/test_load_cell_acquisition.py`
compare against the RES-669 isometric ramp closed forms and inject offset, clipping,
dropped samples and mislabelled units. Engineering calculations only.
