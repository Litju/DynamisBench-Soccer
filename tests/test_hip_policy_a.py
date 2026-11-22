"""Hip squeeze controls based on a sample-aligned isometric ramp (RES-697)."""

import hashlib
import json
from pathlib import Path

import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.engines.nordic_hip import MethodSpec, NordicSpec, SideDeclaration, process
from dynamisbench_soccer.qc.spec import Status
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.reference.motion_models import IsometricRamp
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import Calibration, Quantity, SignalBundle

RAMP = IsometricRamp()
HZ = 1000
LEVER_M = 0.4
LEVER_UNCERTAINTY_M = 0.01
# With knots on the 1 ms grid, trapezoidal integration and constant gain are
# exact in real arithmetic. This bound allows floating-point arithmetic only.
ABS = 1e-8


def _spec(cell: CellSpec | None = None) -> NordicSpec:
    return NordicSpec(
        protocol="hip_squeeze",
        force_basis="raw_calibrated",
        sides=(SideDeclaration(channel="force", side="unilateral"),),
        cells={} if cell is None else {"force": cell},
        lever_arm_m=LEVER_M,
        moment_arm_definition="perpendicular distance from hip joint centre to cuff",
        lever_arm_uncertainty_m=LEVER_UNCERTAINTY_M,
    )


def test_hip_cuff_force_reconstruction() -> None:
    observation = export_case("nordic_hip.hip", scenario=RAMP, hz=HZ).observation
    result = process(observation, _spec(), MethodSpec(max_mean_window_s=0.5))
    side = result.sides["unilateral"]

    assert side.metrics["peak_force_n"] == pytest.approx(2500, rel=0, abs=ABS)
    assert side.metrics["peak_torque_n_m"] == pytest.approx(1000, rel=0, abs=ABS)
    assert side.metrics["max_mean_force_n"] == pytest.approx(2500, rel=0, abs=ABS)
    assert side.metrics["impulse_to_peak_n_s"] == pytest.approx(500, rel=0, abs=ABS)
    assert side.metrics["impulse_record_n_s"] == pytest.approx(2625, rel=0, abs=ABS)
    assert result.bilateral["available"] is False


def test_hip_gain_and_lever_uncertainty_against_ramp() -> None:
    observation = export_case("nordic_hip.hip", scenario=RAMP, hz=HZ).observation
    report = process(observation, _spec()).report()
    assert report["uncertainty"]["peak_force_variants_n"]["gain_-0.010"] == pytest.approx(
        {"unilateral": 2475}, rel=0, abs=ABS
    )
    assert report["uncertainty"]["peak_force_variants_n"]["gain_+0.010"] == pytest.approx(
        {"unilateral": 2525}, rel=0, abs=ABS
    )
    assert report["uncertainty"]["peak_torque_lever_uncertainty_n_m"] == pytest.approx(
        {"unilateral": RAMP.peak_n * LEVER_UNCERTAINTY_M}, rel=0, abs=ABS
    )
    assert (
        report["uncertainty"]["scope"]
        == "conditional sensitivity, not a complete uncertainty budget"
    )


def test_hip_calibration_and_geometry_apply_once() -> None:
    export = export_case("nordic_hip.hip", scenario=RAMP, hz=HZ)
    observation = export.observation
    # Invert the declared bench transfer function F = 500 V - 50.
    channels = [
        channel.model_copy(update={"quantity": Quantity.VOLTAGE, "unit": "V"})
        if channel.name == "force"
        else channel
        for channel in observation.meta.channels
    ]
    acquisition = observation.meta.acquisition
    assert acquisition is not None
    sources = [
        source.model_copy(
            update={"calibration": [], "calibration_unknown_reason": "declared at processing"}
        )
        for source in acquisition.sources
    ]
    voltage = SignalBundle(
        observation.meta.model_copy(
            update={
                "channels": channels,
                "calibration": [],
                "acquisition": acquisition.model_copy(update={"sources": sources}),
            }
        ),
        {**observation.data, "force": (observation["force"] + 50) / 500},
    )
    cell = CellSpec(
        calibration=Calibration(
            date="2000-01-01",
            method="declared affine",
            channel="force",
            gain=500,
            offset=-50,
            input_unit="V",
            output_unit="N",
        )
    )
    metrics = process(voltage, _spec(cell)).sides["unilateral"].metrics
    assert metrics["peak_force_n"] == pytest.approx(2500, rel=0, abs=ABS)
    assert metrics["peak_torque_n_m"] == pytest.approx(1000, rel=0, abs=ABS)
    with pytest.raises(ValueError, match="need a declared calibration"):
        process(voltage, _spec())
    with pytest.raises(ValueError, match="apply twice"):
        process(observation, _spec(cell))

    no_definition = _spec().model_copy(update={"moment_arm_definition": None})
    unavailable = process(observation, no_definition).sides["unilateral"]
    assert "peak_torque_n_m" not in unavailable.metrics
    assert "torque" in unavailable.refusals


def test_hip_cli_hashes_processed_inputs_and_reproduces_report(tmp_path: Path) -> None:
    observation = export_case("nordic_hip.hip", scenario=RAMP, hz=HZ).observation
    stem = tmp_path / "observation"
    parquet, sidecar = io.write(observation, stem)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"spec": _spec().model_dump(mode="json")}))
    out = tmp_path / "report.json"
    args = ["nordic-hip", "process", str(stem), "--config", str(config), "--out", str(out)]

    assert main(args) == 0
    initial = out.read_bytes()
    provenance = json.loads(initial)["provenance"]
    assert provenance == {
        "bundle_id": observation.meta.bundle_id,
        "data_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "validated_metadata_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    assert main(args) == 0
    assert out.read_bytes() == initial

    changed_force = observation["force"].copy()
    changed_force[900] += 1
    changed = SignalBundle(observation.meta, {**observation.data, "force": changed_force})
    io.write(changed, stem)
    assert main(args) == 0
    report = json.loads(out.read_bytes())
    assert report["sides"]["unilateral"]["metrics"]["peak_force_n"] == 2501
    assert report["provenance"]["data_sha256"] != provenance["data_sha256"]
    assert (
        report["provenance"]["validated_metadata_sha256"] != provenance["validated_metadata_sha256"]
    )

    config.write_text(config.read_text() + "\n")
    assert main(args) == 0
    assert json.loads(out.read_bytes())["provenance"]["configuration_sha256"] == (
        hashlib.sha256(config.read_bytes()).hexdigest()
    )
    assert (
        json.loads(out.read_bytes())["provenance"]["configuration_sha256"]
        != provenance["configuration_sha256"]
    )

    parquet.write_bytes(parquet.read_bytes() + b"altered")
    assert main(args) == 2
    assert "sha256 mismatch" in json.loads(out.read_bytes())["error"]


def test_hip_declared_rails_and_undeclared_clipping() -> None:
    observation = export_case("nordic_hip.hip", scenario=RAMP, hz=HZ).observation
    clipped = inject(observation, ContinuousFault(ContinuousKind.CLIPPING, ("force",), 2000))
    for bundle, spec, status, verdict, hits in (
        (observation, _spec(CellSpec(raw_range=(-100, 3000))), Status.PASS, "accept_with_flags", 0),
        (clipped, _spec(CellSpec(raw_range=(-100, 2000))), Status.FAIL, "reject", 841),
        (clipped, _spec(), Status.NOT_APPLICABLE, "accept_with_flags", None),
    ):
        result = process(bundle, spec)
        rails = next(check for check in result.qc if check.rule_id == "LC-RAILS")
        assert rails.status is status
        # The clipped samples span [0.820, 1.660] on the 1 ms grid.
        assert rails.values.get("samples_at_rail") == hits
        assert result.verdict == verdict
        assert result.report()["diagnostic_only"] is (verdict == "reject")


def test_hip_irregular_timing_flag_and_uniform_filter_refusal() -> None:
    observation = export_case("nordic_hip.hip", scenario=RAMP, hz=HZ).observation
    dropped = inject(observation, ContinuousFault(ContinuousKind.DROP, ("force",), stride=3))
    result = process(dropped, _spec())
    timing = next(check for check in result.qc if check.rule_id == "LC-TIMING")
    assert timing.status is Status.FAIL
    assert result.verdict == "accept_with_flags"
    with pytest.raises(ValueError, match="uniform sampling"):
        process(dropped, _spec(), MethodSpec(filter={"kind": "butterworth_lowpass"}))


def test_hip_contradictory_protocol_and_missing_channel_refusals() -> None:
    observation = export_case("nordic_hip.hip", scenario=RAMP, hz=HZ).observation
    with pytest.raises(ValueError, match="contradicts device class"):
        process(observation, _spec().model_copy(update={"protocol": "nordic_eccentric"}))
    missing = SideDeclaration(channel="absent", side="unilateral")
    with pytest.raises(ValueError, match="missing or not a force/voltage channel"):
        process(observation, _spec().model_copy(update={"sides": (missing,)}))


def test_hip_refuses_conflicting_configured_lever() -> None:
    observation = export_case("nordic_hip.hip", scenario=RAMP, hz=HZ).observation
    assert observation.meta.geometry is not None
    assert observation.meta.geometry.lever_arm_m == 0.4
    with pytest.raises(ValueError, match="configured lever arm contradicts bundle geometry"):
        process(observation, _spec().model_copy(update={"lever_arm_m": 0.35}))
    assert process(observation, _spec()).sides["unilateral"].metrics["peak_torque_n_m"] == 1000
