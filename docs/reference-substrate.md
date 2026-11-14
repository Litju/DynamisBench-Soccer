# RES-669 analytical truth substrate

Separate mathematical scenarios support independent numerical verification. They do
not establish physiological validity, manufacturer validity, a calibration certificate,
empirical interchangeability, or qualified AI grading. PR #5 remains a draft.

| Scenario | Implemented truth | Current observation coverage |
| -- | -- | -- |
| Vertical jump | CMJ/SJ/DJ, continuous x/v, exact piecewise acceleration/GRF, impulse, flight and height | Single/dual plate partition, ideal vertical IMU, cable length, contact switching latency |
| Isometric | External force ramp/hold/release, peak/RFD/impulse | Raw load-cell voltage, declared bilateral Nordic/hip-style force partition |
| Eccentric | Prescribed force-angle curve with known break point and perpendicular lever torque | Bilateral force partition; no manufacturer corrections |
| Sprint | Exact exponential x/v/a, bounded numerical gate inverse | Ideal radar cosine velocity, laser range, gate crossing plus latency |
| Planar/COD | Piecewise constant acceleration, continuous field position and velocity | Local position at declared point with latency |
| Rigid body | Planar translation, constant yaw, body-fixed offset with angular velocity and centripetal acceleration | Cable length/count quantization; sensor-frame specific force and yaw gyro, bias and saturation |
| Geodetic | Prescribed WGS84 latitude/longitude/height histories | Coordinate fixes with latency; no unqualified conversion to local metres or Doppler speed |

All eight families and 15 subfamilies have partial analytical J/K evidence. No device
engine is verified. The functions consume separate appropriate scenarios; geodetic
coordinate histories are not claimed to coincide physically with the planar tracks.
The current rigid body has yaw only, not full articulated three-axis orientation.

Sprint mechanics explicitly assume zero aerodynamic drag and constant vertical
support mg: F0=m*Vmax/tau, V0=Vmax, Pmax=F0*Vmax/4. RF=a/sqrt(a²+g²).
`instantaneous_drf_per_m_s` is the derivative dRF/dv, not a fitted DRF regression;
these ideal quantities must not be substituted for a fully specified assessment.

Encoder quantization is nearest integer with half ties upward for nonnegative cable
length. The maximum ideal reconstruction error is half a count. Contact latency has
separate on/off values; observed flight changes by their difference. IMU specific
force is R.T*(a_sensor-g), with the offset's centripetal acceleration included.
No raw manufacturer algorithm or proprietary accumulated-load metric is inferred.

Tests compare closed-form invariants, independent finite differences, explicit known
coordinate cases and quantization bounds. Default float comparisons use 1e-12 absolute
where stated; sprint crossing position error is checked within 1e-11 m for tested
parameter ranges. Seeded Gaussian noise is byte-reproducible under the frozen Linux
lockfile, and does not mutate clean truth. AST import checks prohibit device/assessment
engine imports in references and reference imports in engines.

M2 reader/QC findings are tracked and repaired in RES-705. Shape fixtures remain
physically inconsistent and never verify analytical metric calculations. Exact flat
plateaus can trigger heuristic QC; truth must not be distorted to satisfy a heuristic.

Remaining RES-669 work before acceptance: schema-v2 bundle exporters with category and
provenance per subfamily; complete protocol/continuous/event/position/inertial fault
catalogues; explicit gate beam/body-point geometry; hopping and broader articulated
scenarios; synchronized overlapping-device fixtures; numerical edge-case/tolerance
review. Full GNSS WGS84-to-field/Doppler correspondence requires a declared independently
verified coordinate model. No large datasets or paid model calls were used.

M3 remains 0/8 engineering-verified families. M5 grader qualification and M8 independent
scientific reviews remain later gates; mathematical consistency earns neither.
