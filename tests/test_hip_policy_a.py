"""Hip squeeze controls based on a sample-aligned isometric ramp (RES-697)."""

import pytest

from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.engines.nordic_hip import MethodSpec, NordicSpec, SideDeclaration, process
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.motion_models import IsometricRamp
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
