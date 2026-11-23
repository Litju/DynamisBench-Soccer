# RES-700 GNSS and LPS reference engines

`dbs gnss-lps process` reads one v2 Parquet and JSON observation bundle plus a
producer acquisition declaration. It reconstructs a track for the measured
antenna or tag point. It also reports Doppler speed when GNSS supplies that
channel. A quality failure refuses the complete trial before smoothing or
differentiation.

## Acquisition

Both modalities require an explicit `time` channel in seconds and a binary
`signal_valid` channel. Position and integrity channels must belong to the one
device source. The declaration's `bundle_id` must match the observation metadata.
When a rate is declared in the bundle and acquisition declaration, it is
expressed in samples per source-clock second. The default rate discrepancy is
3% and the default largest step deviation from the median interval is 10%.
Duplicate timestamps, nonmonotonic samples, gaps and position jumps refuse.
The default jump limit is 25 m/s over a single 3D displacement and its mapped
time interval. These limits are engineering options.

- **GNSS.** `latitude` and `longitude` are WGS84 degrees and `height` is
  ellipsoidal metres. The declaration supplies `enu_origin` with
  `latitude_deg`, `longitude_deg` and `ellipsoidal_height_m`. The engine
  converts each fix through ECEF to east, north, up. It expects local tracks
  within 10 km per axis. The WGS84 constants and topocentric axes follow the
  published definitions. [NGA WGS84](https://earth-info.nga.mil/?action=wgs84&dir=wgs84)
  [PROJ topocentric conversion](https://proj.org/en/stable/operations/conversions/topocentric.html)
  Latitude, longitude and height must reference the same declared geodetic frame.
- **GNSS diagnostics.** The producer supplies `hdop` (dimensionless),
  `satellites` and `fix_quality` (integer counts), and `multipath` (binary).
  Admission requires positive HDOP up to 3 by default, at least four satellites
  by default, a zero multipath flag and a supported fix class. Normalize the
  source fix classes to 1 (autonomous), 2 (differential), 4 (RTK fixed), or 5
  (RTK float) before admission. Unknown or dead-reckoned states fail. A
  source-native `doppler_speed` channel in m/s requires a declared
  `doppler_basis` of `horizontal` or `three_dimensional`. Manufacturer
  summaries in the v2 vendor-reported field play no role in computation.
- **LPS.** Position has one `x`, `y` and `z` component in one declared
  field frame, with metres along two orthogonal horizontal axes and local up.
  Set `lps_z_up_frame` to that frame ID to declare the axis convention
  for both positions and anchors.
  Supply at least three horizontally distinct, noncollinear anchors in that
  same frame. `anchor_visibility` maps each anchor ID to a distinct binary
  channel. Each anchor must be visible for every admitted fix. The engine works
  from native field-position solutions and uses the geometry to check the
  declaration. Anchor ranging, localization and per-fix geometry uncertainty
  require separate evidence.

The declaration records `stage`, `native_hz`, `timestamp_basis`,
`latency_s`, and `latency_uncertainty_s`. `native` refers to consecutive,
unaltered receiver fixes or LPS solutions at the acquisition rate. The
`vendor_resampled` and `vendor_processed` stages refuse, as does known
upstream interpolation or resampling declared in bundle provenance. Native
status depends on the producer preserving provenance. A regular clock and
plausible positions cannot establish that undocumented interpolation is absent.

All admitted channels share the same time column and latency declaration. For
receipt timestamps, the engine applies the declared affine clock mapping
and then subtracts the constant receipt latency in reference-clock seconds.
For measurement timestamps, use zero latency correction. Clock drift above
1,000 ppm by default, conflicting drift declarations, and nonzero drift
without sync events spanning the record refuse. Clock and latency uncertainty
bounds are added conservatively and must stay below one native reference
interval. The report retains the original timestamps, mapped timestamps, and
latency-corrected measurement timestamps. Variable latency, spatial covariance,
unflagged multipath, receiver clock metrology, anchor survey and occlusion
accuracy, and sensor-to-body offsets remain unresolved by this calculation.

## Processing

- Process XYZ position and supplied Doppler speed with independently named
  filters `none`, `boxcar5`, or `binomial5`. These offline filters use
  five-sample windows. Position smoothing uses linear extensions at the
  boundaries. Speed smoothing repeats the endpoint speeds. These support
  values are part of the filter definition.
- `three_point` uses three-point differences with actual elapsed times
  including second-order one-sided endpoints. `five_point` uses a centered
  five-point stencil at interior samples on an effectively uniform grid, with
  the three-point method at the two samples nearest each end. Apply the same
  derivative again to velocity for XYZ acceleration. Horizontal positional
  speed and acceleration magnitudes use the horizontal components of the
  respective vectors. Doppler speed change is a scalar derivative in its
  declared speed basis.
- Accumulated distance sums horizontal chords between consecutive processed
  positions and starts at zero. For each threshold and dwell rule, an interval
  qualifies when both endpoint speeds reach the threshold. Keep runs when their
  contiguous elapsed time reaches the dwell and report duration and run count.
  For a qualifying positional run, report horizontal chord distance. For a
  qualifying Doppler run, report trapezoidal distance from the Doppler speed
  in its declared basis. Sampled intervals bound the time resolution.
- Compare peak values under the selected filters, alternative filters and
  every second original sample. The lower-rate comparison retains equal first
  and last timestamps by dropping the final odd-index sample from all
  sensitivity variants when needed. Decimation is a stress test without
  anti-aliasing. It describes rate sensitivity for this record and is available
  with at least nine samples. Five-point rate sensitivity also requires
  uniform grids at both rates over the comparison span. The native result
  and filter comparison remain available if either grid falls outside that
  limit. Acceleration peaks include one-sided boundary
  estimates and can move materially with filtering, sampling, or noise.

## Limits carried by every position report

- Position is the tracked antenna or tag point. The independent rigid-body
  turning control shows that an offset point moves when the translation is
  stationary. There is no sensor-to-body correction.
- LPS starts from producer-solved coordinates. The LPS acquisition contract
  requires surveyed anchors and visibility flags, which establish input
  geometry and recorded visibility only. Ranges, ranging residuals and
  per-sample covariance are outside the observation schema.
- The acquisition contract applies the declared affine clock and fixed
  latency. Absent a variable latency model, the reported timing bounds are
  conditional on those declarations. Metadata and native sample checks
  establish only the documented input stage.
- Registry claims DEV-005 (Linke et al.) and CX-003 cover differences in
  position-derived and Doppler speed and dependence on device, processing,
  rate and thresholds. The engine's threshold distance uses whole sampled
  intervals and its position derivatives depend on boundary support.
- The processing specification above compares offline filters and a
  stride-two subset of the same record. The decimated result supplies
  a sensitivity control within that record.

For example, an acquisition declaration for a GNSS observation with a receipt
clock and horizontal Doppler could be

```json
{
  "bundle_id": "example-gnss-run",
  "stage": "native",
  "native_hz": 18,
  "timestamp_basis": "receipt",
  "latency_s": 0.038,
  "latency_uncertainty_s": 0.001,
  "enu_origin": {"latitude_deg": 40, "longitude_deg": -3, "ellipsoidal_height_m": 0},
  "doppler_basis": "horizontal"
}
```

```sh
dbs gnss-lps process /path/to/observation --declaration /path/to/acquisition.json \
  --position-filter binomial5 --derivative three_point \
  --threshold 3 0 --threshold 3 0.5 --out /path/to/kinematics.json
```

## Verification and qualification

The tests use RES-669 synthetic observations with separate analytical truth
and deterministic fault injection. They check independent closed-form position,
speed, acceleration and distance, WGS84 longitude wrap, clock recovery,
measured-time dwell, Doppler separation, and refusal of corrupted observations.
Both modalities run through the installed wheel.

The GNSS and LPS ledger remains `IN_PROGRESS`. Open requirements include
accepted independent engineering and instrument evidence, audited raw
observations and metrology, sprint assessment profiling in RES-673, and
empirical agreement in RES-703. GNSS Doppler, differentiated GNSS/LPS
position, radar, laser, and timing-gate splits have distinct measurement models.
RES-721 data rights and any empirical validity claims remain outside RES-700.
