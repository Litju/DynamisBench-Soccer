"""Nordic / hip-strength engine versus independent closed forms (RES-697).

Expected values are derived here from the RES-669 eccentric ramp definition: strap force
rises linearly with angle to peak P at the break angle, then falls linearly to zero at the
end angle. With constant descent rate w, peak time = break_angle / w, impulse to peak =
P * t_peak / 2 and total impulse = P * T / 2. The engine never imports the substrate.
"""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines import nordic_hip
from dynamisbench_soccer.engines.nordic_hip import (
    LIMITATIONS,
    MethodSpec,
    NordicSpec,
    SideDeclaration,
    process,
)
from dynamisbench_soccer.qc.spec import Status
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import ChannelSpec, Quantity, SignalBundle

PEAK, BREAK, END, RATE, LEVER = 400.0, math.pi / 4, math.pi / 2, 0.2, 0.4
LEFT_SHARE = 0.4
T_PEAK, T_END = BREAK / RATE, END / RATE
DEFINITION = "perpendicular distance from knee joint centre to strap attachment"


def _spec(**kw: object) -> NordicSpec:
    base: dict[str, object] = {
        "protocol": "nordic_eccentric",
        "force_basis": "raw_calibrated",
        "sides": (
            SideDeclaration(channel="force_left", side="left"),
            SideDeclaration(channel="force_right", side="right"),
        ),
        "lever_arm_m": LEVER,
        "moment_arm_definition": DEFINITION,
    }
    return NordicSpec(**{**base, **kw})  # type: ignore[arg-type]


def _case() -> SignalBundle:
    return export_case("nordic_hip.nordic", hz=1000).observation


def _with_angle(
    bundle: SignalBundle, angle_rad: np.ndarray, name: str = "knee_angle"
) -> SignalBundle:
    """Add a measured angle channel on the same device clock as the force channels."""
    channel = ChannelSpec(name=name, quantity=Quantity.ANGLE, unit="deg")
    acquisition = bundle.meta.acquisition
    assert acquisition is not None
    source = acquisition.sources[0].model_copy(
        update={"channels": [*acquisition.sources[0].channels, name]}
    )
    meta = bundle.meta.model_copy(
        update={
            "channels": [*bundle.meta.channels, channel],
            "acquisition": acquisition.model_copy(update={"sources": [source]}),
        }
    )
    return SignalBundle(meta, {**bundle.data, name: np.degrees(angle_rad)})


def test_peak_torque_impulse_and_asymmetry_match_closed_forms() -> None:
    result = process(_case(), _spec())
    left, right = result.sides["left"].metrics, result.sides["right"].metrics
    for metrics, share in ((left, LEFT_SHARE), (right, 1 - LEFT_SHARE)):
        peak = PEAK * share
        assert metrics["peak_force_n"] == pytest.approx(peak, abs=0.05)
        assert metrics["peak_time_s"] == pytest.approx(T_PEAK, abs=0.002)
        assert metrics["peak_torque_n_m"] == pytest.approx(peak * LEVER, abs=0.02)
        assert metrics["impulse_to_peak_n_s"] == pytest.approx(peak * T_PEAK / 2, rel=2e-4)
        assert metrics["impulse_record_n_s"] == pytest.approx(peak * T_END / 2, rel=2e-4)
    assert result.bilateral["sum_peak_force_n"] == pytest.approx(PEAK, abs=0.1)
    assert result.bilateral["asymmetry_value_percent"] == pytest.approx(
        100 * LEFT_SHARE / (1 - LEFT_SHARE), abs=0.05
    )
    assert result.bilateral["weaker_side"] == "left"
    assert result.verdict in {"accept", "accept_with_flags"}


def test_difference_over_mean_asymmetry_variant_is_distinct() -> None:
    result = process(_case(), _spec(), MethodSpec(asymmetry="difference_over_mean_percent"))
    expected = 100 * (0.6 - 0.4) / 0.5
    assert result.bilateral["asymmetry_value_percent"] == pytest.approx(expected, abs=0.05)
    assert result.bilateral["asymmetry"] == "difference_over_mean_percent"


def test_measured_angle_gives_work_and_peak_force_angle() -> None:
    bundle = _case()
    angle = np.minimum(END, RATE * bundle["time"])
    result = process(
        _with_angle(bundle, angle),
        _spec(angle_channel="knee_angle"),
        MethodSpec(break_point="peak_force_angle"),
    )
    left = result.sides["left"].metrics
    assert left["angle_at_peak_force_deg"] == pytest.approx(45.0, abs=0.06)
    assert left["break_angle_deg"] == pytest.approx(45.0, abs=0.06)
    work = 0.5 * (PEAK * LEFT_SHARE) * LEVER * END
    assert left["eccentric_work_j"] == pytest.approx(work, rel=1e-3)


def test_kinematic_break_point_from_independent_angle_fixture() -> None:
    bundle = _case()
    t = bundle["time"]
    accel, t_break = 1.0, 3.5
    angle = RATE * t + 0.5 * accel * np.clip(t - t_break, 0, None) ** 2
    result = process(
        _with_angle(bundle, angle),
        _spec(angle_channel="knee_angle"),
        MethodSpec(break_point="angular_acceleration", break_velocity_factor=2.0),
    )
    # omega = RATE + accel*(t - t_break) exceeds 2*RATE when t - t_break = RATE / accel.
    expected_deg = math.degrees(RATE * (t_break + RATE / accel) + 0.5 * accel * (RATE / accel) ** 2)
    assert result.sides["left"].metrics["break_angle_deg"] == pytest.approx(expected_deg, abs=0.5)


def test_angle_is_never_recovered_from_force_or_time() -> None:
    result = process(_case(), _spec(), MethodSpec(break_point="peak_force_angle"))
    left = result.sides["left"]
    assert "angle_at_peak_force_deg" not in left.metrics
    assert "never recovered from force" in left.refusals["angle_metrics"]
    assert "measured angle" in left.refusals["break_point"]


def test_kinematic_break_refused_without_controlled_descent() -> None:
    bundle = _case()
    flat = np.full(bundle.meta.n_rows, 0.3)
    result = process(
        _with_angle(bundle, flat),
        _spec(angle_channel="knee_angle"),
        MethodSpec(break_point="angular_acceleration"),
    )
    assert result.sides["left"].metrics["break_angle_deg"] is None
    assert "no positive controlled descent rate" in result.sides["left"].refusals["break_point"]


def test_torque_refused_without_lever_arm_or_definition() -> None:
    no_definition = process(_case(), _spec(moment_arm_definition=None))
    no_lever = process(_case(), _spec(lever_arm_m=None, moment_arm_definition=None))
    for result in (no_definition, no_lever):
        for side in result.sides.values():
            assert "peak_torque_n_m" not in side.metrics
            assert "torque refused" in side.refusals["torque"]
        assert "peak_force_n" in result.sides["left"].metrics


def test_side_ambiguity_and_contradiction_are_refused() -> None:
    with pytest.raises(ValidationError, match="same side"):
        _spec(
            sides=(
                SideDeclaration(channel="force_left", side="left"),
                SideDeclaration(channel="force_right", side="left"),
            )
        )
    with pytest.raises(ValidationError, match="one side only"):
        _spec(
            sides=(
                SideDeclaration(channel="force_left", side="left"),
                SideDeclaration(channel="force_left", side="right"),
            )
        )
    swapped = _spec(
        sides=(
            SideDeclaration(channel="force_left", side="right"),
            SideDeclaration(channel="force_right", side="left"),
        )
    )
    with pytest.raises(ValueError, match="side contradiction"):
        process(_case(), swapped)
    unilateral = _spec(sides=(SideDeclaration(channel="force_left", side="unilateral"),))
    with pytest.raises(ValueError, match="exactly one left and one right"):
        process(_case(), unilateral)


def test_force_basis_is_mandatory_and_carried_through() -> None:
    payload = _spec().model_dump()
    payload.pop("force_basis")
    with pytest.raises(ValidationError, match="force_basis"):
        NordicSpec.model_validate(payload)
    corrected = process(_case(), _spec(force_basis="vendor_corrected"))
    report = corrected.report()
    assert report["force_basis"] == "vendor_corrected"
    assert report["bilateral"]["force_basis"] == "vendor_corrected"  # type: ignore[index]
    assert any("never pooled" in text for text in LIMITATIONS)


def test_protocol_must_match_device_and_geometry_must_be_declared() -> None:
    with pytest.raises(ValueError, match="contradicts device class"):
        process(_case(), _spec(protocol="hip_squeeze"))
    hip = export_case("nordic_hip.hip", hz=1000).observation
    with pytest.raises(ValueError, match="contradicts device class"):
        process(hip, _spec())
    geometry = _case().meta.geometry.model_copy(update={"knee_angle_deg": None})
    stripped = SignalBundle(_case().meta.model_copy(update={"geometry": geometry}), _case().data)
    with pytest.raises(ValueError, match="knee/start angle is undeclared"):
        process(stripped, _spec())


def test_hip_squeeze_metrics_and_geometry_gate() -> None:
    hip = export_case("nordic_hip.hip", hz=1000).observation
    spec = NordicSpec(
        protocol="hip_squeeze",
        force_basis="raw_calibrated",
        sides=(SideDeclaration(channel="force", side="unilateral"),),
        lever_arm_m=0.4,
        moment_arm_definition="hip joint centre to cuff",
    )
    result = process(hip, spec, MethodSpec(max_mean_window_s=0.5))
    side = result.sides["unilateral"].metrics
    assert side["peak_force_n"] == pytest.approx(2500, abs=1e-6)
    assert side["max_mean_force_n"] == pytest.approx(2500, abs=1e-6)  # hold is 0.7 s > 0.5 s
    assert side["peak_torque_n_m"] == pytest.approx(1000, abs=1e-5)
    assert result.bilateral["available"] is False
    geometry = hip.meta.geometry.model_copy(update={"hip_flexion_deg": None})
    stripped = SignalBundle(hip.meta.model_copy(update={"geometry": geometry}), hip.data)
    with pytest.raises(ValueError, match="hip flexion angle"):
        process(stripped, spec)


def test_repetition_detection_finds_each_effort() -> None:
    bundle = export_case("nordic_hip.hip", hz=1000).observation
    force = bundle["force"]
    stacked = np.concatenate([force, force * 0.8, force * 0.9])
    time = np.arange(len(stacked)) / 1000
    meta = bundle.meta.model_copy(update={"n_rows": len(stacked)})
    three = SignalBundle(meta, {"time": time, "force": stacked})
    spec = NordicSpec(
        protocol="hip_squeeze",
        force_basis="raw_calibrated",
        sides=(SideDeclaration(channel="force", side="unilateral"),),
    )
    result = process(three, spec, MethodSpec(repetition="detect", rep_min_separation_s=1.0))
    peaks = [r["peak_force_n"] for r in result.sides["unilateral"].repetitions]
    assert peaks == pytest.approx([2500, 2000, 2250], abs=1e-6)
    assert result.sides["unilateral"].metrics["mean_peak_force_n"] == pytest.approx(2250)
    assert "torque" in result.sides["unilateral"].refusals


def test_corruption_is_flagged_not_hidden() -> None:
    clipped = inject(_case(), ContinuousFault(ContinuousKind.CLIPPING, ("force_left",), 100.0))
    result = process(clipped, _spec())
    assert result.sides["left"].metrics["peak_force_n"] == pytest.approx(100.0, abs=0.05)
    dropped = inject(_case(), ContinuousFault(ContinuousKind.DROP, ("force_left",), stride=3))
    uneven = process(dropped, _spec())
    assert next(r for r in uneven.qc if r.rule_id == "LC-TIMING").status is Status.FAIL
    with pytest.raises(ValueError, match="uniform sampling"):
        process(dropped, _spec(), MethodSpec(filter={"kind": "butterworth_lowpass"}))  # type: ignore[arg-type]


def test_gain_sensitivity_and_lever_uncertainty_are_propagated() -> None:
    spec = _spec(lever_arm_uncertainty_m=0.01)
    result = process(_case(), spec)
    plus = result.uncertainty["peak_force_variants_n"]["gain_+0.010"]  # type: ignore[index]
    assert plus["left"] == pytest.approx(1.01 * result.sides["left"].metrics["peak_force_n"])
    torque_u = result.uncertainty["peak_torque_lever_uncertainty_n_m"]["left"]  # type: ignore[index]
    assert torque_u == pytest.approx(result.sides["left"].metrics["peak_torque_n_m"] * 0.01 / LEVER)


def test_engine_does_not_import_reference_or_claim_muscle_force() -> None:
    tree = ast.parse(Path(nordic_hip.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("dynamisbench_soccer.reference")
    assert any("not hamstring or hip-muscle force" in text for text in LIMITATIONS)


def test_cli_round_trip_and_refusal_exit_codes(tmp_path: Path) -> None:
    case = export_case("nordic_hip.nordic", hz=1000)
    stem = tmp_path / "obs"
    io.write(case.observation, stem)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"spec": json.loads(_spec().model_dump_json())}))
    out = tmp_path / "report.json"
    assert (
        main(["nordic-hip", "process", str(stem), "--config", str(config), "--out", str(out)]) == 0
    )
    report = json.loads(out.read_text())
    assert report["engine"] == "nordic-hip.v1" and report["force_basis"] == "raw_calibrated"
    assert report["sides"]["left"]["metrics"]["peak_force_n"] == pytest.approx(160, abs=0.05)
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"spec": {**json.loads(_spec().model_dump_json()), "protocol": "hip_squeeze"}})
    )
    assert main(["nordic-hip", "process", str(stem), "--config", str(bad), "--out", str(out)]) == 2
    assert "contradicts device class" in json.loads(out.read_text())["error"]
    assert (
        main(["nordic-hip", "process", str(stem), "--config", str(config), "--out", str(config)])
        == 2
    )
