# RES-699: Radar / laser reference engine

Stacked on reviewed RES-669 at bc927bd. This is bounded engineering
verification, not measurement-device validity or a completed M3 qualification.

## Acquisition: two distinct measured quantities

The v2 input is a verified Parquet + JSON bundle containing ONE source:

- Radar: radial Doppler velocity in m/s, with declared aiming point, mounting
  height, device clock, sample rate, geometry, and native/processed stage.
  An axis-projected velocity requires an independently declared cosine angle
  calibration; the algorithm refuses missing, conflicting or near-perpendicular
  angles. Output remains a projection at the declared marker, not COM truth.
- Laser: source-native measured range in m, not speed. The engine filters
  distance first, then differentiates it with a named central or five-point
  finite-difference scheme to yield line-of-sight range rate in m/s. Five-point
  requires a verified uniform native clock; the centered alternative uses
  measured timestamps. No raw optical range can be recovered from processed
  velocity alone.

A manufacturer-processed declaration is distinct from native acquisition:
the output is explicitly labeled as supplied/derived from upstream processing.
The processor refuses to infer or invert proprietary manufacturer filtering or
apply an unqualified extra cosine correction to already processed channels.
The RES-669 synthetic fixture is a known mathematical projection, not a vendor
measurement. No recorded athlete data or proprietary algorithm is copied here.

The schema explicitly records mounting height and aiming point. It does NOT have
a dedicated surveyed distance-behind-start or anatomical body-point offset.
A supplied mount distance is reported, not treated as permission to infer an
angle, COM position or a calibrated correction.

## Clock, integrity, reconstruction and start

Sample timestamps are mapped by declared v2 affine clock alignment. Native
rate and measured rate must agree within the configured admission tolerance.
Missing/duplicate or non-monotonic times, dropout gaps, conflicting clock
metadata, negatives and absurd velocities refuse. Sustained flat clipping
and physically unreasonable kinematic discontinuities also refuse. The
source stream alone cannot distinguish a foreign reflection from a spike,
target switching, or abrupt aiming movement; no missing samples are filled.

The named *offline* filters are none, five-point boxcar and five-point binomial.
Neither is claimed to reproduce any manufacturer processing. Start conventions:
record origin, sustained three-sample velocity threshold, laser-only range-change
threshold, or a separately declared reference-frame trigger. A threshold is an
algorithmic crossing, NOT the athlete's physically measured sprint onset.
The engine does not perform exponential fitting, F-V profiling, gate timing
or manual replacements for unsupported signal corrections.

Sensitivity reports bounded perturbations in start time, declared cosine
angle, and peak velocity across named filter variants. Body point, aiming
drift, and mounting offset corrections are explicitly unidentifiable from
a single line-of-sight signal and report UNSUPPORTED.

## Reproduce against synthetic reference fixtures

First generate the independent fixtures using the existing RES-669 exporter:

    from pathlib import Path
    from dynamisbench_soccer.reference.exporters import export_case
    export_case("radar_laser.radar", hz=47).write(Path("/tmp/res699-radar"))
    export_case("radar_laser.laser", hz=256).write(Path("/tmp/res699-laser"))

Then call the installed or checkout CLI:

    dbs radar-laser process /tmp/res699-radar/observation --stage native --correction declared_fixed_angle --out /tmp/res699-radar/result.json
    dbs radar-laser process /tmp/res699-laser/observation --stage native --derivative five_point --out /tmp/res699-laser/result.json

The test suite checks native-rate and clock recovery, independent closed-form
radar/laser relations, deterministic fault injection, start alternatives,
reflections, clipping, processed versus native stages, tamper refusal and
source/reference import independence.

## Explicit qualification blockers

- Absolute hardware trigger alignment and empirical sample-clock metrology.
- Target tracking, out-of-beam flags and verified equipment calibration.
- Anatomical aiming point motion and time-varying target geometry.
- Manufacturer algorithm documentation and instrument-specific validation.
- Real athlete assessment validation, reliability and cross-device agreement.
- Sprint assessment analysis in RES-673, and family M3 gate in RES-675.

Radar radial velocity, differentiated laser range, GNSS Doppler or coordinate
speed, and beam-gate segment velocity are separate estimands. The family
qualification ledger remains IN_PROGRESS. RES-720 shorts is manufacturer-
processed/resampled, temporally inconsistent across published channels, and
has unresolved data rights: it is neither accepted as native raw optical range
nor used as physical truth. No paid evaluation, merge or qualification
promotion is authorized by this implementation.
