# LPT and encoder reference engine

Run `dbs lpt-encoder process observation --config config.json` on a v2 signal bundle.
The report includes native samples, source calibration and geometry, count resolution,
indexed integrity findings, analysis methods, kinematic traces, and extrema.
Exit 0 means the input was processed, exit 1 means an integrity fault withheld kinematics,
and exit 2 means the declarations or files were invalid.

```json
{
  "cables": [
    {
      "source": "instrument",
      "channel": "cable_length",
      "max_speed_m_s": 4,
      "full_length_m": 3,
      "slack_channel": "slack"
    }
  ],
  "geometry": {
    "attachment_point": "barbell",
    "vertical_line_xy_m": [0, 0]
  },
  "method": {
    "name": "butterworth_then_difference",
    "cutoff_hz": 12,
    "summary_trim_s": 0.25
  },
  "power": {
    "system_mass_kg": 40,
    "rigid_vertical_translation": true
  },
  "sensitivity": {}
}
```

Replace the illustrative hardware limits and geometry with acquisition-specific declarations.
Metre channels accept source-owned affine calibration. For a linear or rotary encoder with
a constant effective conversion, supply `encoder_counts_per_m` in the source geometry.
The engine multiplies `(count - zero_count) / encoder_counts_per_m` by `count_direction`
and adds `initial_length_m` when supplied. Here `initial_length_m` is the absolute
length at the declared zero count or zero metre displacement. Absolute distance channels
already supply the length datum. Variable spool radius, encoder wraps and mechanical backlash
require upstream correction backed by acquisition evidence.

The input attachment label must match the requested point. A single cable with no usable
geometry yields cable travel and its derivatives. Vertical reconstruction uses one of these
models.

- For declared motion along a fixed vertical line, invert the cable range with the surveyed
  anchor and the fixed horizontal coordinates. The point must stay on the upper branch.
  Near anchor height, small range errors can cause large vertical errors or an invalid range.
- For a declared constant angle to upward vertical, divide cable travel by the cosine of
  that angle. This assumes vertical motion and an effectively constant cable direction over
  the interval. Lateral motion or a materially changing angle invalidates the approximation.
- For two cables, supply two absolute range channels ordered by ascending anchor x and
  optionally select each anchor with `anchor_index`. Both cables must attach to the same point.
  The planar model assumes the point lies in the anchors' vertical plane, on the upper side
  of their baseline.

The engine uses source-owned native timestamps when present. For regular single-clock
acquisitions without timestamps it uses the declared nominal grid and reports that assumption.
Dual channels must share a grid or have bracketing clock alignments and a declared
`max_sync_error_s`. Residual row skew plus combined alignment uncertainty must fit the budget
and half the shortest interval across the aligned grids. The dual model pairs corresponding rows.
Index diagnostics cover gaps, speed-bound jumps with one-count allowance, excessive
retraction speed, a declared binary slack indicator and the declared full-extension limit.
A stationary signal alone cannot identify slack. A fault withholds all trial derivatives.

The three methods are `finite_difference` on native intervals, `cubic_spline` with
not-a-knot endpoints, and `butterworth_then_difference`. The last linearly interpolates
to a uniform grid, applies a fourth-order zero-phase Butterworth filter, differentiates,
and evaluates the traces back on the native rows. Summary extrema can exclude a declared
edge interval. Differentiation amplifies quantization, especially for acceleration. The
Butterworth cutoff also applies when that method is included in a sensitivity run.

Sensitivity reports per-series RMSE and peak changes over the baseline summary rows. It
varies differentiation and cutoff, applies a signed elapsed-time scale error, and perturbs
the declared angle or survey geometry. The survey perturbation moves the fixed line along
x or the right anchor along z. A perturbation that violates a model is reported with its error.
These are deterministic comparisons over prescribed perturbations.

The report distinguishes cable travel, vertical bar motion, and vertical motion of other
attachment points, such as a belt or dowel. Body COM motion requires its own reference or
model. The optional power trace is `mass × (9.80665 + vertical acceleration) × vertical velocity`
for a declared rigidly translating mass represented by the observed point. Bar, belt and body
COM may move differently. Force-platform power, effort during free flight, and body COM
power need separate definitions and independent evidence.

## Analytical checks and qualification

The tests use RES-669's separated observation and truth artifacts and independently
specified polynomial, sinusoidal and planar-range controls. All three methods were checked
at 100 and 250 Hz with irregular, quantized encoder timestamps, and on an offset-clock dual
cable bar trajectory. The sine control uses a high-resolution simulated encoder and a
0.3 s edge exclusion. Its regression tolerances are 0.2 mm displacement, 0.0015 m/s velocity,
0.025 m/s² acceleration and 1 W conditional power. These thresholds apply to the fixtures.
Other tests exercise actual v2 reading, missing geometry, wrong attachment metadata,
clock skew and coverage, indexed integrity failures, and analytical or wrong-source rejection.

The `lpt_encoder` family remains `IN_PROGRESS`. Its engineering gate still needs ledger
evidence binding and review, empirical calibration, geometry, timing and integrity evidence,
assessment phase and decision definitions (RES-671), wider independent numerical evidence
(RES-669/695), comparability studies (RES-703), and an execution dossier. Later gates also
need the product implementation evidence (RES-704) and independent scientific review.
No empirical validity or interchangeability claim is supported by these controls.
