"""RES-700 processing checked against independent RES-669 and closed-form truth."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_gnss_lps_acquisition import track_case

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, SpeedThreshold, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import AcquisitionRefusal
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import PositionFault, PositionKind, inject
from dynamisbench_soccer.reference.spatial import PlanarPhase, PlanarTrajectory, RigidBody


@pytest.mark.parametrize(("kind", "hz"), [("gnss", 18), ("lps", 25)])
@pytest.mark.parametrize("derivative", ["three_point", "five_point"])
def test_closed_form_acceleration_speed_distance_and_doppler(
    kind: str, hz: int, derivative
) -> None:
    translation = PlanarTrajectory((PlanarPhase(3, (2, 0)),), initial_velocity_m_s=(1, 0))
    raw = export_case(
        f"gnss_lps.{kind}",
        RigidBody(translation),
        hz=hz,
        clock=DeviceClock(0.21, 160),
        latency_s=0.038,
    )
    case, declaration = track_case(kind, hz=hz, raw=raw)
    result = process(case.observation, declaration, ProcessingOptions(derivative=derivative))
    t, k = case.truth["time"], result.kinematics
    np.testing.assert_allclose(k.position_m[:, 0], t + t**2, atol=4e-9)
    np.testing.assert_allclose(k.horizontal_positional_speed_m_s, 1 + 2 * t, atol=2e-7)
    np.testing.assert_allclose(k.positional_acceleration_m_s2[:, 0], 2, atol=1e-5)
    np.testing.assert_allclose(k.horizontal_positional_acceleration_m_s2, 2, atol=1e-5)
    np.testing.assert_allclose(k.accumulated_horizontal_distance_m, t + t**2, atol=4e-9)
    if kind == "gnss":
        np.testing.assert_array_equal(k.doppler_speed_m_s, case.observation["doppler_speed"])
        np.testing.assert_array_equal(k.doppler_speed_change_m_s2, 0)
        assert result.sensitivity.native.doppler_speed_m_s == 3
    else:
        assert k.doppler_speed_m_s is None
        assert k.doppler_speed_change_m_s2 is None


def test_turning_point_has_vector_acceleration_and_accumulates_chords() -> None:
    body = RigidBody(
        PlanarTrajectory((PlanarPhase(2, (0, 0)),)), yaw_rate_rad_s=1, sensor_offset_m=(2, 0, 0)
    )
    raw = export_case("gnss_lps.lps", body, hz=25)
    case, declaration = track_case("lps", hz=25, latency=0, raw=raw)
    result = process(case.observation, declaration, ProcessingOptions(derivative="five_point"))
    t, k = case.truth["time"], result.kinematics
    interior = slice(4, -4)
    np.testing.assert_allclose(k.horizontal_positional_speed_m_s[interior], 2, atol=2e-7)
    np.testing.assert_allclose(k.horizontal_positional_acceleration_m_s2[interior], 2, atol=4e-7)
    np.testing.assert_allclose(
        k.positional_acceleration_m_s2[interior, 0], -2 * np.cos(t[interior]), atol=4e-7
    )
    chord = 4 * np.sin(0.04 / 2)
    np.testing.assert_allclose(
        k.accumulated_horizontal_distance_m, np.arange(len(t)) * chord, atol=1e-12
    )


def test_three_point_uses_measured_time_and_five_point_refuses_jitter() -> None:
    case, declaration = track_case("lps", hz=10, drift=0, latency=0)
    observation = case.observation
    observation["time"][1::2] += 0.002
    t = observation["time"] - observation["time"][0]
    observation["position_x"][:] = 2 * t + t**2
    observation["position_y"][:] = 0
    result = process(observation, declaration)
    np.testing.assert_allclose(result.kinematics.horizontal_positional_speed_m_s, 2 + 2 * t)
    np.testing.assert_allclose(result.kinematics.horizontal_positional_acceleration_m_s2, 2)
    with pytest.raises(AcquisitionRefusal, match="uniform time"):
        process(observation, declaration, ProcessingOptions(derivative="five_point"))


def test_doppler_threshold_counts_and_integrates_whole_runs_at_measured_dwell() -> None:
    case, declaration = track_case("gnss", hz=10, drift=0, latency=0)
    speed = case.observation["doppler_speed"]
    speed[:] = 0
    speed[0] = 6  # Single sample has no eligible interval.
    speed[3:6] = [4, 6, 4]
    speed[8:10] = 6
    speed[-2:] = 4
    result = process(
        case.observation,
        declaration,
        ProcessingOptions(
            thresholds=(
                SpeedThreshold(threshold_m_s=4, dwell_s=0),
                SpeedThreshold(threshold_m_s=4, dwell_s=0.2),
                SpeedThreshold(threshold_m_s=6, dwell_s=0),
            )
        ),
    )
    low, sustained, high = [row for row in result.thresholds if row.estimand == "doppler_speed"]
    assert (low.runs, sustained.runs, high.runs) == (3, 1, 1)
    assert (low.duration_s, sustained.duration_s, high.duration_s) == pytest.approx((0.4, 0.2, 0.1))
    assert (
        low.integrated_doppler_distance_m,
        sustained.integrated_doppler_distance_m,
        high.integrated_doppler_distance_m,
    ) == pytest.approx((2, 1, 0.6))
    assert all(
        row.integrated_doppler_distance_m is None
        for row in result.thresholds
        if row.estimand == "horizontal_positional_speed"
    )
    assert low.horizontal_path_distance_m is None
    assert result.sensitivity.alternate_filter.doppler_speed_m_s > 0


def test_seeded_position_noise_changes_sensitivity_on_common_support() -> None:
    case, declaration = track_case("lps", hz=25, drift=0)
    observation = inject(
        case.observation,
        PositionFault(PositionKind.NOISE, index=0, count=30, noise_sd_m=0.008, seed=700),
    )
    result = process(
        observation, declaration.model_copy(update={"bundle_id": observation.meta.bundle_id})
    )
    s = result.sensitivity
    assert s.every_second_sample is not None
    assert s.native.reference_rate_hz == pytest.approx(25)
    assert s.every_second_sample.reference_rate_hz == pytest.approx(12.5)
    assert s.alternate_filter.position_filter == "binomial5"
    assert s.comparison_duration_s == pytest.approx(result.acquisition.clock.elapsed_s[::2][-1])
    assert s.alternate_filter.horizontal_positional_acceleration_m_s2 < (
        s.native.horizontal_positional_acceleration_m_s2
    )
    assert s.every_second_sample.horizontal_positional_acceleration_m_s2 != pytest.approx(
        s.native.horizontal_positional_acceleration_m_s2
    )


def test_five_observations_produce_no_rate_sensitivity_and_keep_native_trace() -> None:
    raw = export_case(
        "gnss_lps.lps", RigidBody(PlanarTrajectory((PlanarPhase(0.45, (1, 0)),))), hz=10
    )
    case, declaration = track_case("lps", hz=10, latency=0, raw=raw)
    result = process(case.observation, declaration)
    assert len(result.kinematics.position_m) == 5
    assert result.sensitivity.every_second_sample is None


@pytest.mark.parametrize("kind", ["gnss", "lps"])
def test_cli_roundtrip_and_refusal(kind: str, tmp_path: Path) -> None:
    case, declaration = track_case(kind)
    case.write(tmp_path)
    contract = tmp_path / "acquisition.json"
    contract.write_text(declaration.model_dump_json())
    output = tmp_path / "result.json"
    args = [
        sys.executable,
        "-m",
        "dynamisbench_soccer.cli",
        "gnss-lps",
        "process",
        str(tmp_path / "observation"),
        "--declaration",
        str(contract),
        "--position-filter",
        "boxcar5",
        "--threshold",
        "3",
        "0.2",
        "--out",
        str(output),
    ]
    acceptance = subprocess.run(args, text=True, capture_output=True, check=False, cwd=tmp_path)
    assert acceptance.returncode == 0, acceptance.stderr
    data = output.read_text()
    report = json.loads(data)
    assert report["acquisition"]["device"] == kind
    assert report["acquisition"]["data_sha256"]
    assert report["options"]["position_filter"] == "boxcar5"
    assert report["thresholds"][0]["dwell_s"] == 0.2
    assert report["kinematics"]["accumulated_horizontal_distance_m"][-1] > 0
    assert (report["kinematics"]["doppler_speed_m_s"] is not None) == (kind == "gnss")
    contract.write_text(
        declaration.model_copy(update={"stage": "vendor_resampled"}).model_dump_json()
    )
    refusal = subprocess.run(args, text=True, capture_output=True, check=False, cwd=tmp_path)
    assert refusal.returncode == 2
    assert "resampled" in refusal.stderr
    assert output.read_text() == data


@pytest.mark.parametrize(
    "deviations",
    [
        [-0.9] * 4 + [0] + [0.9] * 4,  # Cropping shifts the median interval.
        [-0.99, -0.99, -0.99, 0, 0, 0, 0.99, 0.99],  # Half-rate median shifts.
    ],
)
def test_five_point_retains_native_result_when_rate_comparison_is_unavailable(
    deviations: list[float],
) -> None:
    duration = (len(deviations) + 0.25) / 10
    raw = export_case(
        "gnss_lps.lps", RigidBody(PlanarTrajectory((PlanarPhase(duration, (1, 0)),))), hz=10
    )
    case, declaration = track_case("lps", hz=10, latency=0, raw=raw)
    t = np.r_[0.0, np.cumsum(0.1 * (1 + 1e-5 * np.array(deviations)))]
    case.observation["time"][:] = t
    case.observation["position_x"][:] = 2 * t + t**2
    result = process(case.observation, declaration, ProcessingOptions(derivative="five_point"))
    assert result.sensitivity.every_second_sample is None
    assert result.sensitivity.rate_unavailable_reason
    assert result.sensitivity.comparison_duration_s == pytest.approx(t[-1])
    assert len(result.kinematics.position_m) == len(t)
    assert result.thresholds


@pytest.mark.parametrize("source", ["observation.parquet", "observation.json", "acquisition.json"])
def test_cli_refuses_hard_link_to_input(source: str, tmp_path: Path, capsys) -> None:
    case, declaration = track_case("lps")
    case.write(tmp_path)
    contract = tmp_path / "acquisition.json"
    contract.write_text(declaration.model_dump_json())
    protected = tmp_path / source
    original = protected.read_bytes()
    alias = tmp_path / "output.json"
    alias.hardlink_to(protected)
    status = main(
        [
            "gnss-lps",
            "process",
            str(tmp_path / "observation"),
            "--declaration",
            str(contract),
            "--out",
            str(alias),
        ]
    )
    assert status == 2
    assert "output path overlaps an input" in capsys.readouterr().err
    assert protected.read_bytes() == original
