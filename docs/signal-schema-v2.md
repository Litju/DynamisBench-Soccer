# Signal schema v2 (RES-693)

The canonical float64 Parquet payload remains unchanged. Metadata schema v2 adds
raw contracts for all eight families and 15 subfamilies. The v1 schema remains
archived; `dbs signals schema --check` checks v2. The v1 Parquet golden hash stays pinned.

| Family | Raw observations | Geometry and calibration |
| -- | -- | -- |
| Single/dual force plates | force by axis and plate | layout, summed-force policy, per-channel gain/offset |
| LPT/encoder | cable distance/displacement or integer counts | attachment, single/dual anchors, angle, counts/m |
| Load cell/isometric rig | raw V or calibrated N | lever arm, compliance, rig description, gain/offset |
| Nordic/hip | force per side | knee/hip angles, lever arm, calibration record |
| Gates/contact | time/distance or timestamp/binary contact state | beam type, height, start/trigger method, latency calibration |
| Radar/laser | radial velocity or range | mount height, aiming point, cosine angle calibration |
| GNSS/LPS | WGS84 lat/lon/height or local xyz | geodetic datum or anchors, declared field frame |
| IMU | framed xyz specific force and angular velocity; optional magnetic field | placement, sensor/segment/field rotation, bias/scale |

Units: N, N*m, m, s, m/s, m/s^2, rad/s, T, V; geodetic angles in deg;
encoder counts use dimensionless `1` and must be integral. Magnetic field is T,
not microtesla. Specific force and acceleration have separate quantities.
No vendor correction, orientation estimator or proprietary load algorithm is inferred.

Acquisition partitions channels among sources, each with device class, clock,
raw format, optional raw hash, geometry, and calibration records or an explicit
missing-calibration reason. Required geometry fields must be populated or named
in `geometry.unknown`. Unknowns describe a gap; they do not qualify a device.
Legacy producers remain readable with an explicit acquisition-unknown reason.
New raw classes require acquisition metadata. `gnss_imu` remains summary-only;
its source cannot be disambiguated by migration. Existing `radar_laser` remains
available for ambiguous historical exports; native producers can declare radar/laser separately.

Clocks declare source and epoch, with optional measured drift. Every non-reference
clock maps directly to the reference using `t_ref = t + offset_s +
(t-origin_s)*drift_ppm*1e-6`. Paired synchronization events must agree within the
recorded uncertainty. Offset is measured at origin; a positive drift means source
intervals expand on the reference timeline. Multi-device producers must align and
resample channels to the bundle's common row/time grid before writing; calibration
and resampling transforms belong in provenance. The schema never silently aligns data.

Sampling records nominal and measured rates and whether sampling is irregular.
Irregular acquisition requires measured `time` in seconds. Frames name sensor,
segment, field, cable or geodetic coordinates; child-to-parent rotations are proper
orthonormal matrices, with explicit parents. A missing rotation remains unknown.
GNSS height datum/antenna position and LPT attachment are explicit measurement
conventions, not automatically COM position.

Calibration records include date/method, optional input/output units, per-channel
gain/offset, bias/scale, cosine angle, uncertainty and reference. Parameters describe
the calibration; no undocumented manufacturer algorithm is executed.

`MIGRATIONS[1]` deep-copies v1 metadata, advances the version, records unknown
acquisition details and appends a provenance transform. It preserves all samples,
data hash, category, device identity, vendor values and existing lineage. Reading an
old bundle migrates metadata in memory; writing emits v2. Unsupported/future versions
fail. Migration does not manufacture geometry, clocks or calibration.

A/B/C are **specified** for every subfamily, not verified. M3 remains 0/8.
The original M2 shape fixtures remain physically inconsistent and cannot verify metrics.
RES-705 tracks reader integrity and QC timing defects; no affected pathway may feed
scientific reference inputs until those safeguards are repaired and reviewed.
