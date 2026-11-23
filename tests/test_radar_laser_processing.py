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
    case = export_case("radar_laser.radar", hz=rate, clock=DeviceClock(0.12, 60))
    result = process(case.observation, _native(correction_method="declared_fixed_angle"))
    assert result.estimand == "radar_declared_axis_projection_velocity"
    assert result.native_quantity == "velocity"
    np.testing.assert_allclose(result.velocity_m_s, case.truth["velocity"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(result.time_reference_s, case.truth["time"], rtol=0, atol=2e-12)
    assert result.sensitivity["cosine_peak_high_m_s"] > result.peak_velocity_m_s


@pytest.mark.parametrize("rate", [47, 100, 256])
@pytest.mark.parametrize("method", ["central", "five_point"])
def test_laser_distance_derivative_recovers_analytical_speed(rate: int, method: str) -> None:
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
    native_radar = export_case("radar_laser.radar").observation
    with pytest.raises(AcquisitionRefusal, match="matching source processing provenance"):
        process(
            native_radar,
            ProcessingOptions(channel_stage=ChannelStage.MANUFACTURER_PROCESSED),
        )
    raw = native_radar.meta.model_dump(mode="json")
    raw["acquisition"]["sources"][0]["raw_format"] = "manufacturer_processed_velocity_csv"
    radar = SignalBundle(
        BundleMeta.model_validate(raw),
        {name: values.copy() for name, values in native_radar.data.items()},
    )
    with pytest.raises(AcquisitionRefusal, match="opaque"):
        process(
            radar,
            ProcessingOptions(
                channel_stage=ChannelStage.MANUFACTURER_PROCESSED,
                correction_method="declared_fixed_angle",
            ),
        )
    result = process(radar, ProcessingOptions(channel_stage=ChannelStage.MANUFACTURER_PROCESSED))
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
    with pytest.raises(AcquisitionRefusal, match=r"discontinuity|negative reconstructed"):
        process(_changed(laser, "distance", 65, 1.1), _native())


def test_smooth_aiming_drift_is_not_improperly_corrected() -> None:
    case = export_case("radar_laser.radar", hz=47)
    altered = inject(
        case.observation,
        ContinuousFault(ContinuousKind.DRIFT, ("velocity",), magnitude=0.25),
    )
    estimated = process(altered, _native(correction_method="declared_fixed_angle"))
    assert (
        estimated.peak_velocity_m_s
        > process(
            case.observation, _native(correction_method="declared_fixed_angle")
        ).peak_velocity_m_s
    )
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
        ContinuousFault(ContinuousKind.NOISE, ("distance",), magnitude=0.00001, seed=9),
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


def test_clipped_native_radar_plateau_refuses_without_inventing_speed() -> None:
    radar = export_case("radar_laser.radar").observation
    clipped = inject(radar, ContinuousFault(ContinuousKind.CLIPPING, ("velocity",), magnitude=3.0))
    with pytest.raises(AcquisitionRefusal, match="flat rail"):
        process(clipped, _native())


def test_start_offset_sensitivity_reports_actual_line_of_sight_speed_at_one_second() -> None:
    radar = export_case("radar_laser.radar", hz=47).observation
    result = process(radar, _native(timing_perturbation_s=0.04))
    sensitivity = result.sensitivity
    minus = sensitivity["los_velocity_at_1s_start_minus_m_s"]
    baseline = sensitivity["los_velocity_at_1s_baseline_m_s"]
    plus = sensitivity["los_velocity_at_1s_start_plus_m_s"]
    assert isinstance(minus, float)
    assert isinstance(baseline, float)
    assert isinstance(plus, float)
    assert 0 < minus < baseline < plus
    assert baseline == pytest.approx(np.interp(1.0, result.time_reference_s, result.velocity_m_s))


def test_stationary_prestart_zeros_are_not_false_clipping() -> None:
    radar = export_case("radar_laser.radar", hz=47).observation
    time = radar["time"]
    # Preserve a genuine standing-zero interval and begin the same known
    # smooth sprint trace at t=12 native samples, with no artificial jump.
    shifted = np.interp(time - time[12], time, radar["velocity"], left=0.0)
    assert np.count_nonzero(shifted[:13]) == 0
    rested = SignalBundle(
        BundleMeta.model_validate(radar.meta.model_dump()),
        {"time": time.copy(), "velocity": shifted},
    )
    result = process(rested, _native())
    assert result.velocity_m_s[0] == 0.0
    assert result.peak_velocity_m_s > 1.0


@pytest.mark.parametrize("angle_deg", [-0.5, 0, 0.5, 10])
def test_radar_cosine_sensitivity_includes_the_closest_angle_to_zero(angle_deg: float) -> None:
    bundle, peak = _radar_at_calibrated_angle(angle_deg)
    report = process(bundle, _native(correction_method="declared_fixed_angle"))
    line_peak = peak * math.cos(math.radians(angle_deg))
    endpoints = [abs(angle_deg - 1), abs(angle_deg + 1)]
    least_angle = 0 if -1 <= angle_deg <= 1 else min(endpoints)
    lower = line_peak / math.cos(math.radians(least_angle))
    upper = line_peak / math.cos(math.radians(max(endpoints)))
    assert report.sensitivity["cosine_peak_low_m_s"] == pytest.approx(lower, rel=0, abs=1e-12)
    assert report.sensitivity["cosine_peak_high_m_s"] == pytest.approx(upper, rel=0, abs=1e-12)
    assert lower <= report.peak_velocity_m_s <= upper


def _radar_at_calibrated_angle(angle_deg: float) -> tuple[SignalBundle, float]:
    case = export_case("radar_laser.radar", hz=100)
    raw = case.observation.meta.model_dump()
    angle = math.radians(angle_deg)
    for record in raw["calibration"] + raw["acquisition"]["sources"][0]["calibration"]:
        if record["cosine_angle_rad"] is not None:
            record["cosine_angle_rad"] = angle
    data = {"time": case.observation["time"], "velocity": case.truth["velocity"] * math.cos(angle)}
    return SignalBundle(BundleMeta.model_validate(raw), data), float(case.truth["velocity"][-1])


def test_radar_projection_uncertainty_refuses_ill_conditioned_perturbations() -> None:
    bundle, peak = _radar_at_calibrated_angle(74.8)
    result = process(bundle, _native(correction_method="declared_fixed_angle"))
    assert result.peak_velocity_m_s == pytest.approx(peak, abs=1e-12, rel=0)
    assert result.sensitivity["cosine_sensitivity"] == (
        "UNSUPPORTED: perturbed angle crosses projection limit"
    )
    assert "cosine_peak_low_m_s" not in result.sensitivity
    assert "cosine_peak_high_m_s" not in result.sensitivity


def test_radar_start_and_filter_sensitivity_against_exponential_velocity() -> None:
    from dynamisbench_soccer.reference.motion_models import MonoExponentialSprint

    model = MonoExponentialSprint()
    radar = export_case("radar_laser.radar", model, hz=100).observation
    result = process(radar, _native(timing_perturbation_s=0.04))
    # The queried times lie exactly on the 100 Hz sample grid.
    times = (0.96, 1.0, 1.04)
    expected = [9 * (1 - math.exp(-t / 1.2)) * math.cos(0.2) for t in times]
    actual = [
        result.sensitivity[f"los_velocity_at_1s_{key}_m_s"]
        for key in ("start_minus", "baseline", "start_plus")
    ]
    assert actual == pytest.approx(expected, rel=0, abs=1e-12)

    duration = (radar.meta.n_rows - 1) / 100
    # Positive smoothing weights preserve monotonicity. Evaluate the final
    # five-point neighborhood with the specified linear endpoint extension.
    first, second, last = [
        9 * (1 - math.exp(-t / 1.2)) * math.cos(0.2)
        for t in (duration - 0.02, duration - 0.01, duration)
    ]
    step = last - second
    samples = (first, second, last, last + step, last + 2 * step)
    box = sum(samples) / 5
    binomial = (
        sum(value * weight for value, weight in zip(samples, (1, 4, 6, 4, 1), strict=True)) / 16
    )
    spread = max(last, box, binomial) - min(last, box, binomial)
    assert spread > 0
    assert result.sensitivity["filter_peak_spread_m_s"] == pytest.approx(spread, rel=0, abs=1e-12)

    no_window = process(
        radar,
        _native(start_method="external_trigger", external_trigger_reference_s=duration - 0.1),
    )
    assert no_window.sensitivity["los_velocity_at_1s"] == (
        "UNSUPPORTED: start perturbation outside observed clock"
    )
    assert "los_velocity_at_1s_baseline_m_s" not in no_window.sensitivity
