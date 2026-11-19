# RES-669 analytical reference substrate

The implementation supplies mathematical verification fixtures for all eight device
families and 15 subfamilies. PR #5 remains unmerged until review and acceptance.
This is engineering reference infrastructure, not physiological validation, empirical
device agreement, calibration certification, AI-grader qualification or M3 completion.

## Generate and inspect artifacts

```python
from pathlib import Path
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.signals import io

case = export_case("radar_laser.radar", hz=47, clock=DeviceClock(0.12, 50))
case.write(Path("/tmp/res669-radar"))
bad = inject(
    case.observation,
    ContinuousFault(
        ContinuousKind.NOISE,
        ("velocity",),
        magnitude=0.1,
        seed=5,
    ),
)
io.write(bad, Path("/tmp/res669-radar/corrupted"))
```

Artifacts are generated on demand; no recorded player data or large artifacts enter
Git. `export_case` accepts the appropriate scenario and native sampling rate, an affine
clock and nonnegative observation latency. Wrong scenario/device combinations reject.
See [reference-verification.md](reference-verification.md) for the evidence matrix.

## Truth, observations and corruptions

- **Truth:** closed-form states, tagged `analytical`, with provenance stage
  `analytical_truth`. A reference device class (`manual_protocol`) identifies this
  mathematical table; it is never a vendor export. Numerical methods are used only
  for explicitly tested inversions/conversions with stated tolerances.
- **Simulated observation:** the relevant instrument projection, tagged `synthetic`
  and stage `simulated_observation`, with full v2 acquisition/frame/geometry/clock
  metadata, prescribed mathematical calibration and a parent truth digest.
- **Corrupted observation:** a copied synthetic observation tagged
  `corrupted_observation`, with the complete typed fault recipe/seed and parent digest.
  Truth, empirical data and vendor-bearing inputs are refused. No-effect requests
  reject: unchanged data cannot acquire an incorrect corruption label.

Arrays and nested metadata are independent between stages. Every non-time channel
has canonical units. Prescribed identity/voltage calibrations are model parameters,
not manufacturer defaults or measured hardware calibrations. The fixed metadata date
`2000-01-01` is the synthetic model epoch, not an acquisition/calibration observation.
Factory jumps prescribe no-arm/`hands_on_hips` conditions; DJ records the input drop
height. Hop protocol IDs record the actual hop count: `prescribed_hops_5` is an explicit
shortened fixture in the `rj_10_5` assessment context, not a completed ten-jump protocol.
Custom unnamed vertical scenarios require an explicit assessment ID; contradictory
named assessments reject. Tracking fixtures use a declared sprint context, with flying
start recorded when their initial COM translation is nonzero.

`Provenance.parents[].sha256` uses `content_hash`: SHA256 of sorted compact JSON
metadata (file `data_sha256` set to null), then ordered little-endian float64 sample
bytes. This binds the complete definition and samples and stays stable after writing.
Parquet SHA256 remains independently recorded/verified by the existing I/O layer.
The complete recipe and common `scenario_sha256` are retained. Sensor metadata is
not an oracle for a manufacturer's proprietary reconstruction.

## Scenarios and instrument projections

| Scenario | Mathematical truth | Observations |
| -- | -- | -- |
| CMJ/SJ/DJ/hops | Piecewise constant acceleration, continuous x/v, exact reaction/impulse, paired complete flights and jump height | Single/dual GRF, cable length/counts, aligned IMU, binary switching events |
| Isometric | Prescribed ramp/hold/release, force/RFD/impulse | Raw V with gain/offset; calibrated isometric/hip force |
| Nordic eccentric | Prescribed angle/force breakpoint and perpendicular knee torque; explicit balancing actuator | Bilateral cell forces, lever geometry; no manufacturer correction |
| Sprint | Exponential x/v/a, bracketed gate inverse; ideal zero-drag mechanics | Radar LOS cosine velocity, laser range, ideal sphere beam crossing |
| Planar/COD | Continuous position/velocity under piecewise constant field acceleration | Local positions and shared spatial sensor trajectories |
| Rigid body | Full constant body-axis angular velocity, proper initial orientation, sensor offset/centripetal acceleration | Geometric LPT/encoder, local/geodetic positions, sensor-frame specific force and gyro |
| Geodetic | Prescribed WGS84 coordinates, or common ENU trajectory converted through ECEF | GNSS fixes; no inferred receiver filtering or satellite Doppler algorithm |

GRF power is `GRF*v`; net power is `m*a*v`. They differ by the gravitational work
rate and are never treated as the same metric. Jump height requires upward takeoff.
Initial drop flight is not paired with a later rebound takeoff. Hops expose every
complete takeoff/landing interval, including final braking mechanics.

Nordic resistance work is the exact triangular force-angle area times lever arm.
The prescribed constant angular rate obeys gravity - knee resistance + balancing
actuator = I*alpha = 0. This explicitly actuated rigid-link experiment does not assert
human muscle capacity, passive fall dynamics or an athlete's break angle.

Rigid rotation uses the matrix exponential/Rodrigues expression for a constant body
angular velocity. Specific force is R.T*(a_sensor-g); gyro is body-frame angular rate.
Sensor acceleration includes the body-offset centripetal term. LPT cable length is
not COM displacement; encoder quantization is nearest count, half ties upward, with
at most half-count reconstruction error. Scalar LPT angle remains explicitly unknown
when it varies; full anchor/attachment geometry is recorded in the scenario/truth.

Sprint F0=m*Vmax/tau, V0=Vmax and Pmax=F0*Vmax/4 assume zero air drag and constant
vertical support mg. RF=a/sqrt(a²+g²); `instantaneous_drf_per_m_s` is dRF/dv, not a
fitted DRF regression. Gate events use first interception of a prescribed spherical
marker at the declared beam height. A missed beam rejects, rather than inventing an
event. This is not a human body-part or commercial timing-gate algorithm.

WGS84 constants follow [NGA](https://earth-info.nga.mil/?action=wgs84&dir=wgs84).
ECEF and ENU conventions follow the primary
[PROJ cart](https://proj.org/en/stable/operations/conversions/cart.html) and
[topocentric documentation](https://proj.org/en/stable/operations/conversions/topocentric.html)
(EPSG 9602/9836; their composition is 9837). Independent tests include the published
PROJ topocentric example. Field origins/heights and ENU offsets are bounded to 10 km;
ECEF inversion rejects the Earth's central region and failed convergence. Height is
above the reference ellipsoid, not sea level. No atmospheric/multipath/satellite error
or GNSS Doppler/position interchangeability is inferred.

## Native rates and synchronization

Each device has a separate bundle at its native Hz. No unequal-rate source is silently
placed on another's grid. The common scenario digest identifies overlapping truth;
compare only physically overlapping scenarios and the declared observation point.

Clock mapping is `t_ref = (1 + drift_ppm*1e-6)*t_device + offset_s`, with two prescribed
exact synchronization pulses. Observations are stamped at receipt: measurement time
is mapped reference time minus the separately recorded latency. Truth uses measurement
time. Sampling metadata reports native device-clock Hz; common-time intervals expand
or contract with drift. Event exports have no sampling rate; their mathematical truth
uses explicitly irregular evaluation times. This verifies synchronization algebra,
not actual hardware synchronization uncertainty.

## Typed fault catalogue

| Family | Operations | Preconditions/conventions |
| -- | -- | -- |
| Continuous | Seeded Gaussian noise, offset, linear drift, clipping, decimation/resampling, dropped samples | Selected continuous channel units; timestamps excluded; resampling changes all rows and declared rate |
| Events/count streams | Missing, duplicate, out-of-order records, trigger timestamp latency | Whole records preserved/changed consistently; invalid chronology remains readable for QC |
| Positions | Dropouts, metre-vector jumps, seeded metre-space noise, interpolated rates | GNSS shifts/noise use local ENU at each fix; interpolation is per reported coordinate channel, deliberately introducing reconstruction artifacts |
| Inertial | Bias, misalignment, saturation, selected specific-force or gyro vector | Active corruption `v_bad=Q*v_clean` keeps original frame labels; Q is proper orthonormal, not a relabelled coordinate system |
| Protocol | Quiet-stance motion, double contact, step-off, L/R swap | Flat baseline/airborne/loaded starts checked; contact and force observations supported; swap preserves total force |

Fault times/windows are in the source timestamp's seconds; exporter physical latency
is in reference seconds. Missing/duplicate/order faults do not rewrite known clock
alignments to hide corruption. Metadata preserves the intended model while provenance
records the fault. Decimation is explicit resampling, not an anti-alias filter claim.
Interpolated rates are explicitly marked in provenance, not treated as new information.

The supplied Gaussian noise, bias, clipping and dropout parameters are chosen test
perturbations. They are not empirical measurement-error distributions or uncertainty
budgets. Wrong paradigms, invalid parameter domains, impossible placements and no-op
faults reject. M2 QC may diagnose applicable faults but is never used to generate or
validate analytical truth. A perfectly flat analytical plateau may trigger a heuristic.

## Verification and disposition

CI runs the original 311-test baseline and the new independent equations, physical
invariants, finite-difference/convergence checks, all-subfamily metamorphic controls,
wrong-input negative controls, native-clock comparisons, complete v2 round trips,
stage-isolation tests and frozen seeded-byte checks. The reference/engine AST import
boundary remains enforced. See the evidence matrix for tolerances and exact test names.

The implementation is ready for independent engineering acceptance; PR #5 stays
unmerged pending acceptance. J/K gain concrete source/test bindings for all 15
subfamilies but remain partial at the device-qualification level: production engine
comparisons, device-specific acceptance and scientific evidence gaps belong to later
issues. All other A–O ratings, owners, blockers, registry gaps and unknown agreement
verdicts are preserved. M3 remains 0/8 engineering-verified families. M5 grader and M8
scientific qualification remain separate. No paid evaluations or sibling-project
changes are part of RES-669.
