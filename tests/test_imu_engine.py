"""IMU engine versus independent RES-669 rigid-body and trajectory truth (RES-701).

Expected values come from the substrate's closed-form motion (R(t) = R0 exp([w]t),
a = a_translation + w x (w x r), specific force = R^T (a - g)) and from hand-derived
bias-growth formulas. The engine never imports the substrate.
"""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines import imu
from dynamisbench_soccer.engines.imu import (
    GRAVITY_M_S2,
    LIMITATIONS,
    ImuSpec,
    MethodSpec,
    process,
)
from dynamisbench_soccer.qc.spec import Status
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import (
    InertialFault,
    InertialKind,
    inject,
)
from dynamisbench_soccer.reference.kinematics import cmj
from dynamisbench_soccer.reference.spatial import PlanarPhase, PlanarTrajectory, RigidBody
from dynamisbench_soccer.registry.models import load as load_registry
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import SignalBundle

OMEGA = (0.2, 0.1, 0.3)


def _rigid(hz: float = 200) -> tuple[RigidBody, SignalBundle]:
    body = RigidBody(
        PlanarTrajectory(
            (PlanarPhase(1, (0, 0)), PlanarPhase(2, (2, 1)), PlanarPhase(1, (-1, 0.5)))
        ),
        sensor_offset_m=(0.1, 0, 0.2),
        angular_velocity_rad_s=OMEGA,
    )
    return body, export_case("imu.inertial", body, hz=hz).observation


def _qc(result: imu.Processed, rule: str) -> Status:
    return next(r for r in result.qc if r.rule_id == rule).status


def _check(result: imu.Processed, rule: str) -> object:
    return next(r for r in result.qc if r.rule_id == rule)


def test_orientation_and_free_acceleration_match_rigid_body_truth() -> None:
    body, bundle = _rigid()
    spec = ImuSpec(initial_orientation=tuple(map(tuple, body.rotation(0))))  # type: ignore[arg-type]
    result = process(bundle, spec)
    assert result.rotation_field_from_sensor is not None
    assert result.free_acceleration_m_s2 is not None
    for k in range(0, len(result.time_s), 40):
        t = float(result.time_s[k])
        np.testing.assert_allclose(
            result.rotation_field_from_sensor[k], body.rotation(t), rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            result.free_acceleration_m_s2[k], body.sensor_state(t)[2], rtol=0, atol=1e-8
        )
    assert result.summary["yaw_observable"] is False
    assert result.verdict in {"accept", "accept_with_flags"}


def test_specific_force_is_not_free_acceleration() -> None:
    body, bundle = _rigid()
    result = process(bundle, ImuSpec(initial_orientation=tuple(map(tuple, body.rotation(0)))))  # type: ignore[arg-type]
    assert result.free_acceleration_m_s2 is not None
    quiet = int(0.5 * 200)
    assert np.linalg.norm(result.specific_force_m_s2[quiet]) > 9  # at rest it senses +g
    assert np.linalg.norm(result.free_acceleration_m_s2[quiet]) < 1.0
    report = result.report()
    assert "specific_force" in report["estimands"] and "free_acceleration" in report["estimands"]  # type: ignore[operator]
    assert any("not free acceleration" in text for text in LIMITATIONS)


def test_integration_recovers_velocity_and_position_given_declared_initial_conditions() -> None:
    # Trapezoid rule errs by at most |jump| * dt / 2 at each acceleration step: here the
    # steps are 2.24 and 3.35 m/s^2, so velocity <= 5.6 * dt / 2 and position <= that * T.
    hz = 1000
    body, bundle = _rigid(hz)
    velocity_bound = 5.6 / hz / 2 * 1.1
    position_bound = velocity_bound * 4
    r0 = tuple(map(tuple, body.rotation(0)))
    p0, v0, _ = body.sensor_state(0)
    spec = ImuSpec(
        initial_orientation=r0,  # type: ignore[arg-type]
        initial_velocity_m_s=tuple(v0),  # type: ignore[arg-type]
        initial_position_m=tuple(p0),  # type: ignore[arg-type]
    )
    result = process(bundle, spec, MethodSpec(integration="declared_initial_velocity"))
    assert result.velocity_m_s is not None and result.position_m is not None
    last = len(result.time_s) - 1
    t_end = float(result.time_s[last])
    p_end, v_end, _ = body.sensor_state(t_end)
    np.testing.assert_allclose(result.velocity_m_s[last], v_end, rtol=0, atol=velocity_bound)
    np.testing.assert_allclose(result.position_m[last], p_end, rtol=0, atol=position_bound)


def test_integration_is_refused_without_declared_initial_conditions() -> None:
    body, bundle = _rigid()
    spec = ImuSpec(initial_orientation=tuple(map(tuple, body.rotation(0))))  # type: ignore[arg-type]
    result = process(bundle, spec, MethodSpec(integration="declared_initial_velocity"))
    assert (
        result.velocity_m_s is None
        and "no declared initial velocity" in result.refusals["integration"]
    )
    with_v = ImuSpec(initial_orientation=spec.initial_orientation, initial_velocity_m_s=(0, 0, 0))
    partial = process(bundle, with_v, MethodSpec(integration="declared_initial_velocity"))
    assert partial.velocity_m_s is not None and partial.position_m is None
    assert "no declared initial position" in partial.refusals["position"]


def test_uncorrected_bias_growth_matches_closed_form_and_zupt_removes_linear_drift() -> None:
    stationary = RigidBody(
        PlanarTrajectory((PlanarPhase(10, (0, 0)),)), angular_velocity_rad_s=(0, 0, 0)
    )
    clean = export_case("imu.inertial", stationary, hz=100).observation
    bias = 0.05
    biased = inject(clean, InertialFault(InertialKind.BIAS, "specific_force", bias=(bias, 0, 0)))
    spec = ImuSpec(
        initial_orientation=((1, 0, 0), (0, 1, 0), (0, 0, 1)), initial_velocity_m_s=(0, 0, 0)
    )
    result = process(biased, spec, MethodSpec(integration="declared_initial_velocity"))
    assert result.velocity_m_s is not None
    t = float(result.time_s[-1])
    assert result.velocity_m_s[-1, 0] == pytest.approx(bias * t, rel=1e-9, abs=0)  # v error = b*t
    assert result.uncertainty["velocity_error_bound_m_s"] == pytest.approx(
        bias * t, rel=0, abs=1e-12
    )
    assert result.uncertainty["position_error_bound_m"] == pytest.approx(
        0.5 * bias * t**2, rel=0, abs=1e-12
    )
    zupt = ImuSpec(
        initial_orientation=spec.initial_orientation,
        initial_velocity_m_s=(0, 0, 0),
        zupt_windows_s=((9.0, 10.0),),
    )
    corrected = process(biased, zupt, MethodSpec(integration="zupt_linear_drift"))
    assert corrected.velocity_m_s is not None
    assert float(np.abs(corrected.velocity_m_s[-1]).max()) < 0.06  # ~ residual within last window
    assert float(np.abs(corrected.velocity_m_s[::50, 0]).max()) < 0.06
    with pytest.raises(ValueError, match="zero-velocity windows"):
        process(
            biased,
            ImuSpec(initial_orientation=spec.initial_orientation, initial_velocity_m_s=(0, 0, 0)),
            MethodSpec(integration="zupt_linear_drift"),
        )


def test_static_tilt_initialises_roll_pitch_but_not_yaw() -> None:
    roll = math.radians(30)
    c, s = math.cos(roll), math.sin(roll)
    r0 = ((1, 0, 0), (0, c, -s), (0, s, c))
    body = RigidBody(
        PlanarTrajectory((PlanarPhase(5, (0, 0)),)),
        angular_velocity_rad_s=(0, 0, 0),
        initial_rotation=r0,
    )
    bundle = export_case("imu.inertial", body, hz=100).observation
    result = process(bundle, ImuSpec(static_window_s=(0.5, 1.5)))
    assert result.rotation_field_from_sensor is not None
    np.testing.assert_allclose(
        result.rotation_field_from_sensor[0], np.array(r0), rtol=0, atol=1e-9
    )
    assert result.free_acceleration_m_s2 is not None
    assert float(np.abs(result.free_acceleration_m_s2).max()) < 1e-9
    assert "static tilt" in str(result.summary["orientation_source"])
    yawed = process(bundle, ImuSpec(static_window_s=(0.5, 1.5), initial_yaw_rad=1.0))
    assert yawed.rotation_field_from_sensor is not None
    up = np.array([0, 0, 1.0])
    np.testing.assert_allclose(
        yawed.rotation_field_from_sensor[0] @ (np.array(r0).T @ up), up, rtol=0, atol=1e-9
    )
    assert not np.allclose(yawed.rotation_field_from_sensor[0], np.array(r0))


def test_orientation_is_refused_when_unobservable() -> None:
    _, bundle = _rigid()
    result = process(bundle)
    assert result.rotation_field_from_sensor is None and result.free_acceleration_m_s2 is None
    assert "unobservable" in result.refusals["orientation"]


def test_gyro_bias_drift_versus_complementary_bound_matches_closed_form() -> None:
    stationary = RigidBody(
        PlanarTrajectory((PlanarPhase(10, (0, 0)),)), angular_velocity_rad_s=(0, 0, 0)
    )
    clean = export_case("imu.inertial", stationary, hz=100).observation
    bias = 0.05
    biased = inject(clean, InertialFault(InertialKind.BIAS, "angular_velocity", bias=(bias, 0, 0)))
    spec = ImuSpec(initial_orientation=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    up = np.array([0, 0, 1.0])

    def tilt(method: MethodSpec) -> float:
        r = process(biased, spec, method).rotation_field_from_sensor
        assert r is not None
        return math.acos(min(1.0, float((r[-1] @ np.array([0, 0, 1.0])) @ up)))

    assert tilt(MethodSpec(orientation="gyro_integration")) == pytest.approx(bias * 9.99, abs=1e-3)
    bounded = tilt(MethodSpec(orientation="complementary", complementary_gain_rad_s=1.0))
    assert bounded == pytest.approx(bias / 1.0, abs=0.01)  # steady state error = bias / gain


def test_declared_calibration_inverts_a_known_forward_model() -> None:
    _, bundle = _rigid()
    truth = np.column_stack([bundle[f"specific_force_{axis}"] for axis in "xyz"])
    angle = math.radians(2)
    m = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    scale, bias = np.array([1.02, 0.99, 1.01]), np.array([0.1, -0.2, 0.05])
    raw = scale * (truth @ m.T + bias)  # independent forward model
    data = dict(bundle.data)
    for i, axis in enumerate("xyz"):
        data[f"specific_force_{axis}"] = raw[:, i]
    corrupted = SignalBundle(bundle.meta, data)
    spec = ImuSpec(
        accel_scale=tuple(scale),  # type: ignore[arg-type]
        accel_bias_m_s2=tuple(bias),  # type: ignore[arg-type]
        accel_misalignment=tuple(map(tuple, m)),  # type: ignore[arg-type]
        calibration_declared=True,
    )
    np.testing.assert_allclose(
        process(corrupted, spec).specific_force_m_s2, truth, rtol=0, atol=1e-9
    )
    wrong = process(corrupted, ImuSpec()).specific_force_m_s2
    assert float(np.abs(wrong - truth).max()) > 0.1
    with pytest.raises(ValidationError, match="invertible"):
        ImuSpec(accel_misalignment=((1, 0, 0), (1, 0, 0), (0, 0, 1)))
    with pytest.raises(ValidationError, match="proper rotation"):
        ImuSpec(initial_orientation=((2, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_sensor_to_segment_placement_is_declared_and_applied() -> None:
    _, bundle = _rigid()
    rotation = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    with pytest.raises(ValueError, match="placement"):
        process(bundle, ImuSpec(mounting_rotation=rotation))
    placed = process(bundle, ImuSpec(mounting_rotation=rotation, placement="tibia, lateral shank"))
    np.testing.assert_allclose(
        placed.specific_force_m_s2[:, 0], -bundle["specific_force_y"], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        placed.specific_force_m_s2[:, 1], bundle["specific_force_x"], rtol=0, atol=1e-9
    )
    assert placed.declarations["placement"] == "tibia, lateral shank"


def test_flight_time_matches_trajectory_and_free_fall_senses_zero() -> None:
    jump = cmj()
    hz = 500
    bundle = export_case("imu.inertial", jump, hz=hz).observation
    spec = ImuSpec(initial_orientation=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    result = process(bundle, spec)
    assert len(result.flights) == 1
    flight = result.flights[0]
    assert flight.duration_s == pytest.approx(jump.flight_time_s(), abs=2 / hz)
    assert flight.takeoff_s == pytest.approx(jump.takeoff_s(), abs=2 / hz)
    middle = int((flight.takeoff_s + flight.landing_s) / 2 * hz)
    assert float(np.linalg.norm(result.specific_force_m_s2[middle])) < 1e-9  # senses nothing
    assert result.free_acceleration_m_s2 is not None
    assert result.free_acceleration_m_s2[middle, 2] == pytest.approx(-GRAVITY_M_S2, abs=1e-9)
    assert flight.landing_peak_m_s2 is not None and flight.landing_peak_m_s2 > 9.8
    for values in result.uncertainty["flight_time_by_threshold_s"].values():  # type: ignore[union-attr]
        assert values[0] == pytest.approx(flight.duration_s, abs=2 / hz)


def test_flight_events_need_minimum_duration() -> None:
    bundle = export_case("imu.inertial", cmj(), hz=500).observation
    assert process(bundle, method=MethodSpec(flight_min_s=5.0)).flights == ()


def test_integrity_faults_are_detected() -> None:
    body, bundle = _rigid()
    spec = ImuSpec(initial_orientation=tuple(map(tuple, body.rotation(0))))  # type: ignore[arg-type]
    keep = np.arange(bundle.meta.n_rows) % 7 != 3  # drop every seventh sample: gaps in time
    gaps = SignalBundle(
        bundle.meta.model_copy(update={"n_rows": int(keep.sum())}),
        {k: v[keep] for k, v in bundle.data.items()},
    )
    gap_result = process(gaps, spec)
    assert _qc(gap_result, "IMU-GAPS") is Status.FAIL and gap_result.verdict == "reject"
    clipped = inject(bundle, InertialFault(InertialKind.SATURATION, "specific_force", limit=5))
    sat = process(clipped, ImuSpec(accel_range_m_s2=5.0))
    assert _qc(sat, "IMU-SAT-ACCE") is Status.FAIL and sat.verdict == "reject"
    assert _qc(process(bundle, spec), "IMU-SAT-ACCE") is Status.NOT_APPLICABLE
    assert _qc(process(bundle, spec, MethodSpec(spike_limit_m_s2=9.0)), "IMU-SPIKE") is Status.FAIL
    assert _qc(process(bundle, ImuSpec(sync_uncertainty_s=0.05)), "IMU-SYNC") is Status.FAIL
    scaled_time = SignalBundle(bundle.meta, {**bundle.data, "time": bundle["time"] * 1.05})
    assert _qc(process(scaled_time), "IMU-RATE") is Status.FAIL


def test_static_gyro_bias_and_slip_are_flagged() -> None:
    quiet = RigidBody(PlanarTrajectory((PlanarPhase(6, (0, 0)),)), angular_velocity_rad_s=(0, 0, 0))
    clean = export_case("imu.inertial", quiet, hz=100).observation
    spec = ImuSpec(static_window_s=(0.5, 1.5), end_static_window_s=(4.5, 5.5))
    ok = process(clean, spec)
    assert _qc(ok, "IMU-GYROBIAS") is Status.PASS and _qc(ok, "IMU-SLIP") is Status.PASS
    biased = inject(clean, InertialFault(InertialKind.BIAS, "angular_velocity", bias=(0.1, 0, 0)))
    assert _qc(process(biased, spec), "IMU-GYROBIAS") is Status.FAIL
    angle = math.radians(20)
    rot = np.array(
        [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
    )
    data = {k: v.copy() for k, v in clean.data.items()}
    f = np.column_stack([data[f"specific_force_{a}"] for a in "xyz"])
    t = clean["time"]
    f[t >= 3.0] = f[t >= 3.0] @ rot.T  # sensor slips at 3 s, gyro sees nothing
    for i, axis in enumerate("xyz"):
        data[f"specific_force_{axis}"] = f[:, i]
    slipped = process(SignalBundle(clean.meta, data), spec)
    check = _check(slipped, "IMU-SLIP")
    assert check.status is Status.FAIL  # type: ignore[attr-defined]
    assert check.values["inconsistency_rad"] == pytest.approx(angle, rel=0, abs=1e-6)  # type: ignore[attr-defined]


def test_wrong_device_and_missing_channels_are_refused() -> None:
    force_plate = export_case("force_platform.single", hz=200).observation
    with pytest.raises(ValueError, match="does not accept device class"):
        process(force_plate)
    _, bundle = _rigid()
    meta = bundle.meta.model_copy(
        update={"channels": [c for c in bundle.meta.channels if c.name != "angular_velocity_z"]}
    )
    data = {k: v for k, v in bundle.data.items() if k != "angular_velocity_z"}
    with pytest.raises(ValueError):
        process(SignalBundle(meta, data))


def test_engine_never_imports_reference_substrate() -> None:
    tree = ast.parse(Path(imu.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("dynamisbench_soccer.reference")


def test_cli_round_trip(tmp_path: Path) -> None:
    body, bundle = _rigid()
    stem = tmp_path / "imu"
    io.write(bundle, stem)
    config = tmp_path / "config.json"
    spec = ImuSpec(initial_orientation=tuple(map(tuple, body.rotation(0))))  # type: ignore[arg-type]
    config.write_text(json.dumps({"spec": json.loads(spec.model_dump_json())}))
    out = tmp_path / "report.json"
    assert main(["imu", "process", str(stem), "--config", str(config), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["engine"] == "imu.v1" and report["summary"]["yaw_observable"] is False
    assert main(["imu", "process", str(stem), "--config", str(config), "--out", str(config)]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"spec": {"mounting_rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]]}}))
    assert main(["imu", "process", str(stem), "--config", str(bad), "--out", str(out)]) == 2
    assert "placement" in json.loads(out.read_text())["error"]


@pytest.mark.parametrize("orientation", ["gyro_integration", "complementary"])
@pytest.mark.parametrize("range_m_s2", [5.0, 30.0])
def test_imu_retains_sourced_physical_and_numerical_limits(
    orientation: str, range_m_s2: float
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_registry(root / "registry")
    claim = next(claim for claim in registry.claims if claim.id == "DEV-002")
    assert claim.sources == ["nielsen2019"] and "nielsen2019" in registry.sources
    assert "sensor position displacement and COM displacement" in claim.statement
    assert "Registry claim DEV-002 (Nielsen et al.)" in (root / "docs/imu-engine.md").read_text()

    _, bundle = _rigid()
    result = process(
        bundle,
        ImuSpec(initial_orientation=imu.IDENTITY, accel_range_m_s2=range_m_s2),
        MethodSpec(orientation=orientation),
    )
    expected_topics = (
        "Specific force is not free acceleration",
        "cannot observe heading",
        "sustained acceleration can bias roll/pitch even within that tolerance",
        "position error b*t^2/2",
        "Sensor-to-segment placement is a declared rotation",
        "Flight events from the specific-force magnitude",
        "clipping cannot be undone",
        "Time-synchronisation uncertainty is a declaration",
        "no device, placement or physiological qualification",
    )
    assert len(result.limitations) == len(expected_topics)
    assert all(topic in row for topic, row in zip(expected_topics, result.limitations, strict=True))
    report = json.loads(result.report_json())
    assert report["limitations"] == list(result.limitations)
    assert report["diagnostic_only"] is (range_m_s2 == 5)
    assert result.summary["yaw_observable"] is False


def test_imu_complementary_gravity_gate_admits_biased_tilt_under_acceleration() -> None:
    trajectory = PlanarTrajectory((PlanarPhase(2, (2, 0)),))
    bundle = export_case("imu.inertial", RigidBody(trajectory), hz=100).observation
    result = process(
        bundle, ImuSpec(initial_orientation=imu.IDENTITY), MethodSpec(orientation="complementary")
    )
    assert math.hypot(2, 9.80665) - 9.80665 < 0.5

    # In this one-axis case the scalar cross product gives a signed sine of the
    # difference between the acceleration-implied tilt and the current tilt.
    assumed_tilt = math.atan2(2, 9.80665)
    angle = 0.0
    for _ in range(bundle.meta.n_rows - 1):
        angle += 0.01 * math.sin(assumed_tilt - angle)
    assert result.rotation_field_from_sensor is not None
    estimated = math.acos(float(result.rotation_field_from_sensor[-1, 2, 2]))
    assert angle > 0.1  # The true orientation is fixed, so this is a false tilt.
    assert estimated == pytest.approx(angle, rel=0, abs=1e-12)


def test_injected_fixed_imu_misalignment_is_conditional_on_declared_calibration() -> None:
    body = RigidBody(PlanarTrajectory((PlanarPhase(2, (0, 0)),)))
    clean = export_case("imu.inertial", body, hz=100).observation
    angle = 0.2
    rotation = (
        (1.0, 0.0, 0.0),
        (0.0, math.cos(angle), -math.sin(angle)),
        (0.0, math.sin(angle), math.cos(angle)),
    )
    mistilted = inject(clean, InertialFault(InertialKind.MISALIGNMENT, rotation=rotation))
    spec = ImuSpec(
        initial_orientation=imu.IDENTITY,
        static_window_s=(0.2, 0.8),
        end_static_window_s=(1.2, 1.8),
        accel_range_m_s2=20,
        gyro_range_rad_s=3,
    )
    nominal = process(clean, spec)
    assert nominal.verdict == "accept"
    affected = process(mistilted, spec)
    assert affected.verdict == "accept"
    # A constant wrong sensor calibration gives the same tilt at both windows.
    # The slip detector therefore passes while the acceleration is biased.
    assert _qc(affected, "IMU-SLIP") is Status.PASS
    expected = np.array([0, -9.80665 * math.sin(angle), 9.80665 * (math.cos(angle) - 1)])
    assert affected.free_acceleration_m_s2 is not None
    np.testing.assert_allclose(
        affected.free_acceleration_m_s2 - expected[None, :], 0, rtol=0, atol=1e-12
    )
    corrected = process(
        mistilted,
        spec.model_copy(update={"accel_misalignment": rotation, "calibration_declared": True}),
    )
    assert corrected.verdict == "accept"
    np.testing.assert_allclose(corrected.free_acceleration_m_s2, 0, rtol=0, atol=1e-12)
