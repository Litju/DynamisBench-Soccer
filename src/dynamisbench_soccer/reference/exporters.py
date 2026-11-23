"""Schema-v2 analytical truth and simulated instrument exports (RES-669).

No vendor values, empirical calibration or athlete validity are inferred. Each native
clock has its own rectangular bundle. Truth time is measurement time; observation
time is the device timestamp at receipt (physical latency is recorded separately).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.coordinates import LocalFrame
from dynamisbench_soccer.reference.kinematics import Trajectory, cmj
from dynamisbench_soccer.reference.motion_models import IsometricRamp, MonoExponentialSprint
from dynamisbench_soccer.reference.observations import (
    laser_distance_m,
    load_cell_voltage,
    lpt_cable_length_m,
    radar_radial_velocity_m_s,
    spherical_gate_trigger_s,
)
from dynamisbench_soccer.reference.spatial import (
    GeodeticTrack,
    NordicMechanics,
    PlanarPhase,
    PlanarTrajectory,
    RigidBody,
)
from dynamisbench_soccer.reference.spatial_observations import (
    bilateral_force_n,
    cable_length_m,
    contact_events,
    dual_plate_force_n,
    encoder_counts,
    imu_body_observation,
)
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import (
    CANONICAL_UNIT,
    Acquisition,
    AcquisitionSource,
    BundleKind,
    BundleMeta,
    Calibration,
    ChannelSpec,
    Clock,
    Device,
    DeviceClass,
    FloatArray,
    Frame,
    Geometry,
    Parent,
    Protocol,
    Provenance,
    Quantity,
    Sampling,
    Session,
    Side,
    SignalBundle,
    Subject,
)

DEVICES = {
    "force_platform.single": DeviceClass.FORCE_PLATE,
    "force_platform.dual": DeviceClass.FORCE_PLATE,
    "lpt_encoder.lpt": DeviceClass.LPT,
    "lpt_encoder.encoder": DeviceClass.ENCODER,
    "load_cell_isometric.load_cell": DeviceClass.LOAD_CELL,
    "load_cell_isometric.isometric_rig": DeviceClass.ISOMETRIC_DYNAMOMETER,
    "nordic_hip.nordic": DeviceClass.NORDIC_DYNAMOMETER,
    "nordic_hip.hip": DeviceClass.HIP_DYNAMOMETER,
    "timing_contact.gates": DeviceClass.TIMING_GATES,
    "timing_contact.contact": DeviceClass.CONTACT_MAT,
    "radar_laser.radar": DeviceClass.RADAR,
    "radar_laser.laser": DeviceClass.LASER,
    "gnss_lps.gnss": DeviceClass.GNSS,
    "gnss_lps.lps": DeviceClass.LPS,
    "imu.inertial": DeviceClass.IMU,
}
Scenario = (
    Trajectory | IsometricRamp | MonoExponentialSprint | RigidBody | NordicMechanics | GeodeticTrack
)
TRUTH_STAGE = "RES-669 stage=analytical_truth"
OBSERVATION_STAGE = "RES-669 stage=simulated_observation"
CORRUPTED_STAGE = "RES-669 stage=corrupted_observation"


def content_hash(bundle: SignalBundle) -> str:
    """Canonical metadata + ordered little-endian sample bytes, excluding file hash."""
    raw = bundle.meta.model_dump(mode="json")
    raw["data_sha256"] = None
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for values in bundle.data.values():
        digest.update(values.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExportedCase:
    truth: SignalBundle
    observation: SignalBundle

    def write(self, directory: Path) -> None:
        """On-demand artifacts only; returns through the existing verified I/O API."""
        io.write(self.truth, directory / "truth")
        io.write(self.observation, directory / "observation")


def default_scenario(subfamily: str) -> Scenario:
    if subfamily not in DEVICES:
        raise ValueError(f"unsupported reference subfamily {subfamily!r}")
    if subfamily.startswith("force_platform") or subfamily == "timing_contact.contact":
        return cmj()
    if subfamily.startswith("load_cell_isometric") or subfamily == "nordic_hip.hip":
        return IsometricRamp()
    if subfamily == "nordic_hip.nordic":
        return NordicMechanics()
    if subfamily.startswith("radar_laser") or subfamily == "timing_contact.gates":
        return MonoExponentialSprint()
    return RigidBody(
        PlanarTrajectory((PlanarPhase(4, (2, 0)),)),
        sensor_offset_m=(0.1, 0, 0.2),
        angular_velocity_rad_s=(0.2, 0.1, 0.3),
    )


def _duration(scenario: Scenario) -> float:
    if isinstance(scenario, Trajectory):
        return scenario.total_duration_s
    if isinstance(scenario, RigidBody):
        return scenario.translation.duration_s
    if isinstance(scenario, (IsometricRamp, NordicMechanics)):
        return scenario.duration_s
    return 5.0


def _truth(scenario: Scenario, times: FloatArray) -> dict[str, tuple[Quantity, FloatArray]]:
    data: dict[str, tuple[Quantity, FloatArray]] = {}
    if isinstance(scenario, Trajectory):
        states = [scenario.state(float(t)) for t in times]
        for name, q, vals in (
            ("height", Quantity.POSITION, [s.displacement_m for s in states]),
            ("velocity", Quantity.VELOCITY, [s.velocity_m_s for s in states]),
            ("acceleration", Quantity.ACCELERATION, [s.acceleration_m_s2 for s in states]),
            ("grf", Quantity.FORCE, [s.vertical_grf_n for s in states]),
        ):
            data[name] = (q, np.array(vals, dtype=np.float64))
    elif isinstance(scenario, IsometricRamp):
        data["force"] = (Quantity.FORCE, np.array([scenario.force_n(float(t)) for t in times]))
    elif isinstance(scenario, NordicMechanics):
        eccentric_states = np.array([scenario.state(float(t)) for t in times])
        for i, (name, q) in enumerate(
            (
                ("angle", Quantity.ANGLE),
                ("resistance", Quantity.TORQUE),
                ("gravity_torque", Quantity.TORQUE),
                ("actuator_torque", Quantity.TORQUE),
            )
        ):
            data[name] = (
                q,
                eccentric_states[:, i] * 180 / math.pi if i == 0 else eccentric_states[:, i],
            )
        data["force"] = (
            Quantity.FORCE,
            np.array([scenario.curve.force_n(float(t)) for t in times]),
        )
    elif isinstance(scenario, MonoExponentialSprint):
        for name, q, fn in (
            ("position", Quantity.POSITION, scenario.position_m),
            ("velocity", Quantity.VELOCITY, scenario.velocity_m_s),
            ("acceleration", Quantity.ACCELERATION, scenario.acceleration_m_s2),
        ):
            data[name] = (q, np.array([fn(float(t)) for t in times]))
    elif isinstance(scenario, RigidBody):
        rigid_states = [scenario.sensor_state(float(t)) for t in times]
        for k, (prefix, q) in enumerate(
            (
                ("position", Quantity.POSITION),
                ("velocity", Quantity.VELOCITY),
                ("acceleration", Quantity.ACCELERATION),
            )
        ):
            for i, axis in enumerate("xyz"):
                data[f"{prefix}_{axis}"] = (q, np.array([s[k][i] for s in rigid_states]))
        rotations = np.array([scenario.rotation(float(t)) for t in times])
        for i in range(3):
            for j in range(3):
                data[f"rotation_{i}{j}"] = (Quantity.DIMENSIONLESS, rotations[:, i, j])
    else:
        fixes = np.array([scenario.position(float(t)) for t in times])
        for i, (name, q) in enumerate(
            (
                ("latitude", Quantity.LATITUDE),
                ("longitude", Quantity.LONGITUDE),
                ("height", Quantity.POSITION),
            )
        ):
            data[name] = (q, fixes[:, i])
    return data


def _bundle(
    subfamily: str,
    stage: Literal["truth", "observation"],
    spec: str,
    times: FloatArray,
    channels: list[ChannelSpec],
    data: dict[str, FloatArray],
    geometry: Geometry,
    kind: BundleKind,
    hz: float | None,
    clock: DeviceClock,
    test_id: str,
    calibration: list[Calibration],
    frames: list[Frame],
    protocol: Protocol,
) -> SignalBundle:
    is_truth = stage == "truth"
    device = Device(
        device_class=DeviceClass.MANUAL_PROTOCOL if is_truth else DEVICES[subfamily],
        model="RES-669 mathematical reference" if is_truth else "RES-669 ideal observation",
    )
    identifier = hashlib.sha256(spec.encode()).hexdigest()[:16]
    sampling = (
        Sampling(nominal_hz=hz, measured_hz=hz, t0_s=float(times[0])) if hz is not None else None
    )
    clocks = [Clock(id="reference", source="host", epoch="prescribed scenario t=0")]
    if not is_truth:
        clocks.append(
            Clock(
                id="device",
                source="device",
                epoch="affine scenario clock",
                drift_ppm=clock.drift_ppm,
            )
        )
    source = AcquisitionSource(
        id="analytical" if is_truth else "instrument",
        device=device,
        clock="reference" if is_truth else "device",
        channels=list(data),
        geometry=geometry,
        calibration=calibration,
        calibration_unknown_reason="mathematical source; not empirical instrument calibration"
        if not calibration
        else None,
        raw_format="RES-669 deterministic float64 generator",
    )
    meta = BundleMeta(
        bundle_id=f"res669-{subfamily}-{stage}-{identifier}",
        kind=kind,
        test_id=test_id,
        data_category=DataCategory.ANALYTICAL if is_truth else DataCategory.SYNTHETIC,
        subject=Subject(pseudonym="ANALYTICAL-NOT-ATHLETE"),
        session=Session(session_id="RES-669", date="2000-01-01"),
        device=device,
        protocol=protocol,
        sampling=sampling,
        channels=channels,
        n_rows=len(times),
        geometry=geometry,
        calibration=calibration,
        acquisition=Acquisition(
            clocks=clocks,
            reference_clock="reference",
            sources=[source],
            frames=frames,
            alignments=[]
            if is_truth
            else [clock.alignment(max(1.0, float(clock.reference_time(times)[-1])))],
        ),
        acquisition_unknown_reason=None,
        provenance=Provenance(
            created_by="RES-669-reference@1",
            transforms=[
                TRUTH_STAGE if is_truth else OBSERVATION_STAGE,
                "spec=" + spec,
                "scenario_sha256=" + str(json.loads(spec)["scenario_sha256"]),
                "parent hashes use canonical content_hash, not Parquet-only SHA256",
            ],
        ),
    )
    return SignalBundle(BundleMeta.model_validate(meta.model_dump()), data)


def export_case(
    subfamily: str,
    scenario: Scenario | None = None,
    *,
    hz: float = 100,
    clock: DeviceClock | None = None,
    latency_s: float = 0,
    test_id: str | None = None,
) -> ExportedCase:
    """Export one native-rate observation with separate independently defined truth.

    Geometry/calibration constants are prescribed ideal fixtures, fully recorded in
    metadata/spec. They are not manufacturer defaults or measured calibrations.
    """
    clock = DeviceClock() if clock is None else clock
    if subfamily not in DEVICES:
        raise ValueError("unsupported reference subfamily")
    if not math.isfinite(hz) or hz <= 0 or not math.isfinite(latency_s) or latency_s < 0:
        raise ValueError("positive finite native Hz and nonnegative finite latency required")
    scenario = default_scenario(subfamily) if scenario is None else scenario
    device = DEVICES[subfamily]
    allowed: tuple[type[object], ...]
    if device is DeviceClass.FORCE_PLATE or device is DeviceClass.CONTACT_MAT:
        allowed = (Trajectory,)
    elif device in {DeviceClass.LPT, DeviceClass.ENCODER, DeviceClass.IMU}:
        allowed = (Trajectory, RigidBody)
    elif device in {
        DeviceClass.LOAD_CELL,
        DeviceClass.ISOMETRIC_DYNAMOMETER,
        DeviceClass.HIP_DYNAMOMETER,
    }:
        allowed = (IsometricRamp,)
    elif device is DeviceClass.NORDIC_DYNAMOMETER:
        allowed = (NordicMechanics,)
    elif device in {DeviceClass.RADAR, DeviceClass.LASER, DeviceClass.TIMING_GATES}:
        allowed = (MonoExponentialSprint,)
    elif device is DeviceClass.GNSS:
        allowed = (RigidBody, GeodeticTrack)
    else:
        allowed = (RigidBody,)
    if not isinstance(scenario, allowed):
        raise ValueError("scenario does not observe this device's measurand")
    duration = _duration(scenario)
    step = (1 + clock.drift_ppm * 1e-6) / hz
    n = math.floor(duration / step) + 1
    if not 2 <= n <= 1_000_000:
        raise ValueError("sample grid requires 2..1000000 rows")
    reference = np.arange(n, dtype=np.float64) * step
    event = device in {DeviceClass.CONTACT_MAT, DeviceClass.TIMING_GATES}
    spec = json.dumps(
        dict(
            subfamily=subfamily,
            model_version=1,
            scenario_type=type(scenario).__name__,
            scenario=asdict(scenario),
            hz=hz,
            clock=asdict(clock),
            latency_s=latency_s,
            ideal_parameters=dict(
                left_share=0.4,
                volts_per_n=0.001,
                offset_v=0.2,
                counts_per_m=1000,
                anchor_m=[-1, 0, 0],
                radar_angle_rad=0.2,
                laser_offset_m=1,
                geodetic_origin=[40, -3, 0],
                gate_distances_m=[5, 10, 20, 30],
                gate_beam_height_m=1,
                gate_marker_center_height_m=1,
                gate_marker_radius_m=0.1,
            ),
        ),
        sort_keys=True,
        allow_nan=False,
    )
    observation: dict[str, tuple[Quantity, FloatArray]] = {}
    geometry = Geometry()
    calibration: list[Calibration] = []
    requested_test = test_id
    if isinstance(scenario, Trajectory):
        if (
            requested_test is not None
            and scenario.assessment_id is not None
            and requested_test != scenario.assessment_id
        ):
            raise ValueError("assessment contradicts the named analytical scenario")
        test_id = requested_test or scenario.assessment_id
        if test_id is None:
            raise ValueError("unnamed trajectory requires an explicit test_id")
    elif isinstance(scenario, IsometricRamp):
        test_id = "hip_add_abd" if device is DeviceClass.HIP_DYNAMOMETER else "imtp"
    elif isinstance(scenario, NordicMechanics):
        test_id = "nhe"
    else:
        test_id = "sprint_gates" if device is DeviceClass.TIMING_GATES else "sprint_fv"
    if requested_test is not None and requested_test != test_id:
        if not isinstance(scenario, (RigidBody, GeodeticTrack)):
            raise ValueError("assessment contradicts the analytical scenario")
        test_id = requested_test
    if isinstance(scenario, Trajectory) and device in {
        DeviceClass.FORCE_PLATE,
        DeviceClass.CONTACT_MAT,
    }:
        if device is DeviceClass.CONTACT_MAT:
            events = np.array(contact_events(scenario))
            reference = events[:, 0]
            observation["contact"] = (Quantity.DIMENSIONLESS, events[:, 1])
            geometry = Geometry(
                trigger_method="ideal contact on/off; receipt latency recorded in spec"
            )
        elif subfamily.endswith("dual"):
            forces = np.array([dual_plate_force_n(scenario, float(t), 0.4) for t in reference])
            observation = {
                "fz_left": (Quantity.FORCE, forces[:, 0]),
                "fz_right": (Quantity.FORCE, forces[:, 1]),
            }
            geometry = Geometry(
                plate_layout="dual_side_by_side", summed_force_policy="sum_fz_all_plates"
            )
        else:
            observation["fz"] = (
                Quantity.FORCE,
                np.array([scenario.state(float(t)).vertical_grf_n for t in reference]),
            )
            geometry = Geometry(plate_layout="single")
    elif device in {DeviceClass.LPT, DeviceClass.ENCODER}:
        if isinstance(scenario, RigidBody):
            lengths = np.array([cable_length_m(scenario, float(t), (-1, 0, 0)) for t in reference])
        elif isinstance(scenario, Trajectory):
            lengths = np.array(
                [
                    lpt_cable_length_m(scenario, float(t), anchor_x_m=-1, anchor_z_m=0)
                    for t in reference
                ]
            )
        else:
            raise AssertionError("validated scenario")
        geometry = Geometry(
            attachment_point="prescribed sensor offset"
            if isinstance(scenario, RigidBody)
            else "COM",
            cable_anchors_m=[(-1, 0, 0)],
            encoder_counts_per_m=1000,
            unknown=["cable_angle_rad"],
        )  # Angle varies; full 3D anchor/point geometry is in truth.
        if device is DeviceClass.ENCODER:
            observation["counts"] = (
                Quantity.COUNT,
                np.array([encoder_counts(float(v), 1000) for v in lengths], dtype=np.float64),
            )
        else:
            observation["cable_length"] = (Quantity.DISTANCE, lengths)
    elif isinstance(scenario, IsometricRamp):
        geometry = Geometry(
            rig_description="prescribed rigid zero-compliance rig",
            rig_compliance_m_per_n=0,
            lever_arm_m=0.4,
            hip_flexion_deg=45,
        )
        if device is DeviceClass.LOAD_CELL:
            observation["voltage"] = (
                Quantity.VOLTAGE,
                np.array(
                    [
                        load_cell_voltage(scenario, float(t), volts_per_n=0.001, offset_v=0.2)
                        for t in reference
                    ]
                ),
            )
        else:
            observation["force"] = (
                Quantity.FORCE,
                np.array([scenario.force_n(float(t)) for t in reference]),
            )
        calibration = [
            Calibration(
                date="2000-01-01",
                method="prescribed ideal voltage=0.001*force+0.2; not empirical calibration",
                channel="voltage" if device is DeviceClass.LOAD_CELL else "force",
                input_unit="N" if device is DeviceClass.LOAD_CELL else "V",
                output_unit="V" if device is DeviceClass.LOAD_CELL else "N",
                gain=0.001 if device is DeviceClass.LOAD_CELL else 1000,
                offset=0.2 if device is DeviceClass.LOAD_CELL else -200,
            )
        ]
    elif isinstance(scenario, NordicMechanics):
        forces = np.array(
            [bilateral_force_n(scenario.curve.force_n(float(t)), 0.4) for t in reference]
        )
        observation = {
            "force_left": (Quantity.FORCE, forces[:, 0]),
            "force_right": (Quantity.FORCE, forces[:, 1]),
        }
        geometry = Geometry(
            lever_arm_m=scenario.curve.lever_arm_m,
            knee_angle_deg=90,
            rig_description="actuator-balanced prescribed eccentric descent",
        )
    elif isinstance(scenario, MonoExponentialSprint):
        if event:
            distances = np.array([5.0, 10.0, 20.0, 30.0])
            reference = np.array(
                [
                    spherical_gate_trigger_s(
                        scenario, float(d), beam_height_m=1, center_height_m=1, radius_m=0.1
                    )
                    for d in distances
                ]
            )
            observation["distance_m"] = (Quantity.DISTANCE, distances)
            geometry = Geometry(
                gate_height_m=1,
                beam_type="single",
                trigger_method="first intersection of ideal radius 0.1m marker; center height 1m",
            )
        elif device is DeviceClass.RADAR:
            observation["velocity"] = (
                Quantity.VELOCITY,
                np.array(
                    [
                        radar_radial_velocity_m_s(scenario, float(t), angle_rad=0.2)
                        for t in reference
                    ]
                ),
            )
            geometry = Geometry(
                mounting_height_m=1,
                aiming_point="prescribed marker; fixed LOS cosine angle 0.2 rad",
            )
            calibration = [
                Calibration(
                    date="2000-01-01",
                    method="prescribed LOS projection; no cosine correction applied",
                    cosine_angle_rad=0.2,
                )
            ]
        else:
            observation["distance"] = (
                Quantity.DISTANCE,
                np.array([laser_distance_m(scenario, float(t), offset_m=1) for t in reference]),
            )
            geometry = Geometry(
                mounting_height_m=1, aiming_point="marker on laser axis; initial range 1m"
            )
    elif device is DeviceClass.IMU:
        if isinstance(scenario, RigidBody):
            samples = [imu_body_observation(scenario, float(t)) for t in reference]
        elif isinstance(scenario, Trajectory):
            samples = [
                (
                    np.array(
                        [0, 0, scenario.state(float(t)).acceleration_m_s2 + scenario.gravity_m_s2]
                    ),
                    np.zeros(3),
                )
                for t in reference
            ]
        else:
            raise AssertionError("validated scenario")
        for k, (prefix, q) in enumerate(
            (
                ("specific_force", Quantity.SPECIFIC_FORCE),
                ("angular_velocity", Quantity.ANGULAR_VELOCITY),
            )
        ):
            for i, axis in enumerate("xyz"):
                observation[f"{prefix}_{axis}"] = (q, np.array([s[k][i] for s in samples]))
        geometry = Geometry(
            sensor_placement="declared rigid-body offset"
            if isinstance(scenario, RigidBody)
            else "COM, aligned with field"
        )
    elif device in {DeviceClass.LPS, DeviceClass.GNSS}:
        if isinstance(scenario, RigidBody):
            positions = np.array([scenario.sensor_state(float(t))[0] for t in reference])
            if device is DeviceClass.LPS:
                for i, axis in enumerate("xyz"):
                    observation[f"position_{axis}"] = (Quantity.POSITION, positions[:, i])
                geometry = Geometry(
                    lps_anchors_m={"a": (-10, -10, 0), "b": (10, -10, 0), "c": (0, 10, 0)}
                )
            else:
                fixes = np.array([LocalFrame().to_geodetic(p) for p in positions])
        elif isinstance(scenario, GeodeticTrack):
            fixes = np.array([scenario.position(float(t)) for t in reference])
        else:
            raise AssertionError("validated scenario")
        if device is DeviceClass.GNSS:
            for i, (name, q) in enumerate(
                (
                    ("latitude", Quantity.LATITUDE),
                    ("longitude", Quantity.LONGITUDE),
                    ("height", Quantity.POSITION),
                )
            ):
                observation[name] = (q, fixes[:, i])
            geometry = Geometry(
                geodetic_datum="WGS84",
                sensor_placement="prescribed antenna point; height above ellipsoid",
            )
    else:
        raise AssertionError("all supported devices covered")
    if event and len(reference) < 2:
        raise ValueError("event export requires at least two switching/crossing events")
    spec = json.dumps({**json.loads(spec), "test_id": test_id}, sort_keys=True, allow_nan=False)
    scenario_payload = {"scenario_type": type(scenario).__name__, "scenario": asdict(scenario)}
    scenario_digest = hashlib.sha256(
        json.dumps(scenario_payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    spec = json.dumps(
        {**json.loads(spec), "scenario_sha256": scenario_digest}, sort_keys=True, allow_nan=False
    )
    protocol_suffix = ""
    if isinstance(scenario, Trajectory) and scenario.assessment_id == "rj_10_5":
        protocol_suffix = f".prescribed_hops_{len(scenario.flight_intervals())}"
    protocol = Protocol(
        protocol_id=f"res669.ideal.{test_id}{protocol_suffix}.v1",
        arm_swing="hands_on_hips" if isinstance(scenario, Trajectory) else "not_applicable",
        drop_height_m=scenario.prescribed_drop_height_m
        if isinstance(scenario, Trajectory)
        else None,
        action="adduction" if device is DeviceClass.HIP_DYNAMOMETER else None,
        start_mode=("flying" if any(scenario.translation.initial_velocity_m_s) else "standing")
        if isinstance(scenario, RigidBody)
        else "standing"
        if isinstance(scenario, MonoExponentialSprint)
        else "not_applicable",
    )
    truth_data = _truth(scenario, reference)
    truth_time = reference.copy()
    observed_time = clock.device_time(reference + latency_s)

    def columns(
        payload: dict[str, tuple[Quantity, FloatArray]], time: FloatArray, is_truth: bool
    ) -> tuple[list[ChannelSpec], dict[str, FloatArray]]:
        name = "t_s" if event and not is_truth else "time"
        channels = [ChannelSpec(name=name, quantity=Quantity.TIME, unit="s")]
        data = {name: time.copy()}
        for col, (q, values) in payload.items():
            axis: Literal["x", "y", "z"] | None = None
            if col.endswith("_x"):
                axis = "x"
            if col.endswith("_y"):
                axis = "y"
            if col.endswith("_z"):
                axis = "z"
            force_plate = not is_truth and device is DeviceClass.FORCE_PLATE
            if force_plate:
                axis = "z"
            channels.append(
                ChannelSpec(
                    name=col,
                    quantity=q,
                    unit=CANONICAL_UNIT[q],
                    axis=axis,
                    side=Side.LEFT
                    if col.endswith("_left")
                    else Side.RIGHT
                    if col.endswith("_right")
                    else Side.NONE,
                    plate=(2 if col.endswith("_right") else 1) if force_plate else None,
                    frame="sensor"
                    if not is_truth and device is DeviceClass.IMU and q is not Quantity.TIME
                    else "field"
                    if axis
                    else None,
                )
            )
            data[col] = np.array(values, dtype=np.float64, copy=True)
        return channels, data

    frames = [
        Frame(
            id="field",
            kind="field",
            definition="ENU at WGS84 40deg N, 3deg W, h=0; jump z up",
        ),
        Frame(
            id="sensor",
            kind="sensor",
            definition="R(t) body-to-field in rigid truth; aligned for vertical truth",
            parent="field",
        ),
    ]
    truth_channels, truth_arrays = columns(truth_data, truth_time, True)
    # Event truth has irregular evaluation points but remains a mathematical timeseries.
    truth_hz = 1 / step if not event else (len(reference) - 1) / float(reference[-1] - reference[0])
    truth = _bundle(
        subfamily,
        "truth",
        spec,
        truth_time,
        truth_channels,
        truth_arrays,
        Geometry(),
        BundleKind.TIMESERIES,
        truth_hz,
        DeviceClock(),
        test_id,
        [],
        frames,
        protocol,
    )
    if event:
        raw = truth.meta.model_dump()
        raw["sampling"]["irregular"] = True
        truth = SignalBundle(BundleMeta.model_validate(raw), truth.data)
    channels, arrays = columns(observation, observed_time, False)
    if not calibration:
        calibration = [
            Calibration(
                date="2000-01-01",
                method="prescribed mathematical identity, not empirical calibration",
                channel=c.name,
                input_unit=c.unit,
                output_unit=c.unit,
                gain=1,
                offset=0,
            )
            for c in channels
            if c.quantity is not Quantity.TIME
        ]
    if device is DeviceClass.GNSS:
        frames = [
            *frames,
            Frame(
                id="wgs84",
                kind="geodetic",
                definition="WGS84 latitude, longitude and ellipsoidal height",
            ),
        ]
        for channel in channels:
            if channel.name in {"latitude", "longitude", "height"}:
                channel.frame = "wgs84"
    observation_bundle = _bundle(
        subfamily,
        "observation",
        spec,
        observed_time,
        channels,
        arrays,
        geometry,
        BundleKind.EVENTS if event else BundleKind.TIMESERIES,
        None if event else hz,
        clock,
        test_id,
        calibration,
        frames,
        protocol,
    )
    observation_bundle.meta.provenance.parents = [
        Parent(bundle_id=truth.meta.bundle_id, sha256=content_hash(truth))
    ]
    return ExportedCase(truth, observation_bundle)
