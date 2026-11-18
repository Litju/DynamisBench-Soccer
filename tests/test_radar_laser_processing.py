"""RES-699 analytical recovery, named processing and adversarial refusals."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines.radar_laser import ProcessingOptions, process
from dynamisbench_soccer.engines.radar_laser_acquisition import (
    AcquisitionRefusal,
    ChannelStage,
)
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle


def _native(**kwargs):  # type: ignore[no-untyped-def]
    return ProcessingOptions(channel_stage=ChannelStage.NATIVE, **kwargs)


@pytest.mark.parametrize("rate", [47, 100, 256])
def test_radar_recovers_independent_exact_velocity_at_native_rate(rate: int) -> None:
    case = export_case(
        "radar_laser.radar", hz=rate, clock=DeviceClock(0.12, 60)
    )
    result = process(case.observation, _native(correction_method="declared_fixed_angle"))
    assert result.estimand == "radar_declared_axis_projection_velocity"
    assert result.native_quantity == "velocity"
    np.testing.assert_allclose(result.velocity_m_s, case.truth["velocity"], atol=2e-12)
    np.testing.assert_allclose(
        result.time_reference_s, case.truth["time"], atol=2e-12
    )
    assert result.sensitivity["cosine_peak_high_m_s"] > result.peak_velocity_m_s


@pytest.mark.parametrize("rate", [47, 100, 256])
@pytest.mark.parametrize("method", ["central", "five_point"])
def test_laser_distance_derivative_recovers_analytical_speed(
    rate: int, method: str
) -> None:
    case = export_case("radar_laser.laser", hz=rate)
    result = process(case.observation, _native(derivative_method=method))
    assert result.native_quantity == "distance"
    assert result.estimand == "laser_line_of_sight_range_rate"
    error = np.max(np.abs(np.asarray(result.velocity_m_s) - case.truth["velocity"]))
    assert error < 0.004
    assert "body_point_sensitivity" in result.sensitivity


def test_start_modes_are_distinct_and_refuse_missing_triggers() -> None:
    radar = export_case("radar_laser.radar", hz=47).observation
    record = process(radar, _native())
    threshold = process(radar, _native(start_method="velocity_threshold"))
    external = process(
        radar,
        _native(start_method="external_trigger", external_trigger_reference_s=0.03),
    )
    assert record.start_reference_s == pytest.approx(0)
    assert threshold.start_reference_s > 0
    assert external.start_reference_s == pytest.approx(0.03)
    laser = export_case("radar_laser.laser", hz=256).observation
    change = process(laser, _native(start_method="range_change", range_change_m=0.05))
    assert change.start_reference_s > 0
    with pytest.raises(AcquisitionRefusal, match="laser-only"):
        process(radar, _native(start_method="range_change"))
    with pytest.raises(AcquisitionRefusal, match="finite declared"):
        process(radar, _native(start_method="external_trigger"))
    with pytest.raises(AcquisitionRefusal, match="no sustained start"):
        process(radar, _native(threshold_m_s=17, start_method="velocity_threshold"))


def test_unqualified_cosine_correction_refuses_laser_and_processed_radar() -> None:
    laser = export_case("radar_laser.laser", hz=100).observation
    with pytest.raises(AcquisitionRefusal, match="calibration missing"):
        process(laser, _native(correction_method="declared_fixed_angle"))
    radar = export_case("radar_laser.radar").observation
    with pytest.raises(AcquisitionRefusal, match="opaque"):
        process(
            radar,
            ProcessingOptions(
                channel_stage=ChannelStage.MANUFACTURER_PROCESSED,
                correction_method="declared_fixed_angle",
            ),
        )
    result = process(
        radar, ProcessingOptions(channel_stage=ChannelStage.MANUFACTURER_PROCESSED)
    )
    assert result.estimand == "manufacturer_processed_radar_velocity_as_supplied"


def _changed(bundle: SignalBundle, column: str, index: int, bump: float) -> SignalBundle:
    data = {k: v.copy() for k, v in bundle.data.items()}
    data[column][index] += bump
    return SignalBundle(BundleMeta.model_validate(bundle.meta.model_dump()), data)


def test_second_athlete_reflection_and_target_switch_refuse_without_repair() -> None:
    radar = export_case("radar_laser.radar", hz=47).observation
    with pytest.raises(AcquisitionRefusal, match="discontinuity"):
        process(_changed(radar, "velocity", 35, 3.5), _native())
    laser = export_case("radar_laser.laser", hz=256).observation
    with pytest.raises(AcquisitionRefusal, match="discontinuity"):
        process(_changed(laser, "distance", 65, 1.1), _native())


def test_smooth_aiming_drift_is_not_improperly_corrected() -> None:
    case = export_case("radar_laser.radar", hz=47)
    altered = inject(
        case.observation,
        ContinuousFault(
            ContinuousKind.DRIFT, ("velocity",), magnitude=0.25
        ),
    )
    estimated = process(altered, _native(correction_method="declared_fixed_angle"))
    assert estimated.peak_velocity_m_s > process(
        case.observation, _native(correction_method="declared_fixed_angle")
    ).peak_velocity_m_s
    assert "UNSUPPORTED" in str(estimated.sensitivity["aiming_drift_correction"])


def test_filter_variants_are_named_and_report_sensitivity() -> None:
    laser = export_case("radar_laser.laser", hz=256).observation
    results = [
        process(laser, _native(filter_method=variant))
        for variant in ("none", "boxcar5", "binomial5")
    ]
    assert len({r.filter_method for r in results}) == 3
    assert all(r.sensitivity["filter_peak_spread_m_s"] > 0 for r in results)
    assert all(r.derivative_method == "five_point" for r in results)


def test_noise_in_distance_differentiation_is_not_doppler_velocity() -> None:
    case = export_case("radar_laser.laser", hz=256)
    noisy = inject(
        case.observation,
        ContinuousFault(
            ContinuousKind.NOISE, ("distance",), magnitude=0.00001, seed=9
        ),
    )
    unfiltered = process(noisy, _native(filter_method="none"))
    smoothed = process(noisy, _native(filter_method="binomial5"))
    actual = case.truth["velocity"]
    rmse_none = math.sqrt(np.mean((np.asarray(unfiltered.velocity_m_s) - actual) ** 2))
    rmse_smooth = math.sqrt(np.mean((np.asarray(smoothed.velocity_m_s) - actual) ** 2))
    assert rmse_smooth < rmse_none


def test_no_reference_imports_in_engine_files() -> None:
    root = Path(__file__).resolve().parents[1] / "src/dynamisbench_soccer/engines"
    for module in ("radar_laser.py", "radar_laser_acquisition.py"):
        tree = ast.parse((root / module).read_text())
        assert not any(
            "dynamisbench_soccer.reference" in ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        )
