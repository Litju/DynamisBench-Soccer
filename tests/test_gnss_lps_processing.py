"""RES-700 processing checked against independent RES-669 and closed-form truth."""

from __future__ import annotations

import hashlib
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
from dynamisbench_soccer.registry.models import load as load_registry
from dynamisbench_soccer.signals import io


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


@pytest.mark.parametrize("derivative", ["three_point", "five_point"])
@pytest.mark.parametrize(
    ("position_filter", "second_moment"), [("none", 0), ("boxcar5", 2), ("binomial5", 1)]
)
def test_lps_processing_variants_and_reported_sensitivity(
    derivative: str, position_filter: str, second_moment: int
) -> None:
    hz = 25
    # On x=0.75t each symmetric filter and both derivative stencils are exact,
    # including linear endpoint extension and every-second-sample comparison.
    linear = RigidBody(
        PlanarTrajectory((PlanarPhase(3.2, (0, 0)),), initial_velocity_m_s=(0.75, 0))
    )
    raw = export_case("gnss_lps.lps", linear, hz=hz)
    case, declaration = track_case("lps", hz=hz, latency=0, raw=raw)
    options = ProcessingOptions(derivative=derivative, position_filter=position_filter)
    result = process(case.observation, declaration, options)
    assert result.sensitivity.comparison_duration_s == pytest.approx(3.2, abs=1e-12, rel=0)
    assert result.sensitivity.every_second_sample is not None
    for peak, rate in (
        (result.sensitivity.native, 25),
        (result.sensitivity.alternate_filter, 25),
        (result.sensitivity.every_second_sample, 12.5),
    ):
        assert peak.reference_rate_hz == pytest.approx(rate, rel=0, abs=1e-9)
        assert peak.horizontal_positional_speed_m_s == pytest.approx(0.75, rel=0, abs=2e-12)
        assert peak.horizontal_positional_acceleration_m_s2 == pytest.approx(0, rel=0, abs=2e-10)

    # The second moment of each symmetric kernel shifts x=t+t² by μ₂h².
    quadratic = RigidBody(PlanarTrajectory((PlanarPhase(3, (2, 0)),), initial_velocity_m_s=(1, 0)))
    raw = export_case("gnss_lps.lps", quadratic, hz=hz)
    case, declaration = track_case("lps", hz=hz, latency=0, raw=raw)
    result = process(case.observation, declaration, options)
    t = case.truth["time"][6:-6]
    kinematics = result.kinematics
    np.testing.assert_allclose(
        kinematics.position_m[6:-6, 0], t + t**2 + second_moment / hz**2, rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        kinematics.horizontal_positional_speed_m_s[6:-6], 1 + 2 * t, rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        kinematics.positional_acceleration_m_s2[6:-6, 0], 2, rtol=0, atol=2e-10
    )

    # All comparisons end at t=2.96 s. With slope extension, the endpoint
    # velocity is 6.92 - h(3w1 + 5w2) for the two positive-offset weights.
    alternate = "binomial5" if position_filter == "none" else "none"
    endpoint_loss = {"none": 0, "boxcar5": 8 / 5, "binomial5": 17 / 16}
    acceleration_overshoot = {"none": 0, "boxcar5": 1 / 30, "binomial5": 17 / 576}
    sensitivity = result.sensitivity
    assert sensitivity.comparison_duration_s == pytest.approx(2.96, rel=0, abs=1e-12)
    assert sensitivity.every_second_sample is not None
    for peak, filt, rate in (
        (sensitivity.native, position_filter, 25),
        (sensitivity.alternate_filter, alternate, 25),
        (sensitivity.every_second_sample, position_filter, 12.5),
    ):
        assert peak.position_filter == filt
        assert peak.reference_rate_hz == pytest.approx(rate, rel=0, abs=1e-9)
        expected_speed = 6.92 - endpoint_loss[filt] / rate
        assert peak.horizontal_positional_speed_m_s == pytest.approx(
            expected_speed, rel=0, abs=2e-12
        )
        # For the nested five-point derivative, the boundary perturbations
        # peak four rows in for boxcar5 and three rows in for binomial5.
        expected_acceleration = 2 + (
            acceleration_overshoot[filt] if derivative == "five_point" else 0
        )
        assert peak.horizontal_positional_acceleration_m_s2 == pytest.approx(
            expected_acceleration, rel=0, abs=2e-10
        )


def test_lps_positional_threshold_recovers_known_interval_and_refuses_short_dwell() -> None:
    hz = 25
    body = RigidBody(PlanarTrajectory((PlanarPhase(3, (2, 0)),), initial_velocity_m_s=(1, 0)))
    raw = export_case("gnss_lps.lps", body, hz=hz)
    case, declaration = track_case("lps", hz=hz, latency=0, raw=raw)
    options = ProcessingOptions(
        thresholds=(
            SpeedThreshold(threshold_m_s=3.01, dwell_s=1.95),
            SpeedThreshold(threshold_m_s=3.01, dwell_s=1.97),
            SpeedThreshold(threshold_m_s=6.99, dwell_s=0),
        )
    )
    passed, too_short, isolated = process(case.observation, declaration, options).thresholds
    # v=1+2t crosses 3.01 m/s between samples at 1.00 and 1.04 s.
    assert passed.estimand == "horizontal_positional_speed"
    assert passed.runs == 1
    assert passed.duration_s == pytest.approx(1.96, rel=0, abs=1e-12)
    assert passed.horizontal_path_distance_m == pytest.approx(9.8784, rel=0, abs=2e-12)
    assert passed.integrated_doppler_distance_m is None
    for refused in (too_short, isolated):
        assert refused.runs == 0
        assert refused.duration_s == 0
        assert refused.horizontal_path_distance_m == 0


@pytest.mark.parametrize(("kind", "channel"), [("gnss", "height"), ("lps", "position_z")])
def test_position_engines_reproduce_seeded_inputs_and_bind_the_source_hash(
    kind: str, channel: str, tmp_path: Path
) -> None:
    stem = tmp_path / "observation"
    config = tmp_path / "declaration.json"
    out = tmp_path / "report.json"
    fault = PositionFault(PositionKind.NOISE, index=4, count=6, noise_sd_m=0.001, seed=700)
    args = ["gnss-lps", "process", str(stem), "--declaration", str(config), "--out", str(out)]
    replay = []
    for _ in range(2):
        trial, declaration = track_case(kind)
        observation = inject(trial.observation, fault)
        parquet, sidecar = io.write(observation, stem)
        config.write_text(
            declaration.model_copy(
                update={"bundle_id": observation.meta.bundle_id}
            ).model_dump_json()
        )
        assert main(args) == 0
        result = out.read_bytes()
        assert (
            json.loads(result)["acquisition"]["data_sha256"]
            == hashlib.sha256(parquet.read_bytes()).hexdigest()
        )
        replay.append((parquet.read_bytes(), sidecar.read_bytes(), result))
    assert replay[0] == replay[1]

    data = observation[channel]
    data[7] += 0.005
    io.write(observation, stem)
    assert main(args) == 0
    changed = json.loads(out.read_bytes())
    assert (
        changed["acquisition"]["data_sha256"]
        != json.loads(replay[0][2])["acquisition"]["data_sha256"]
    )
    assert changed["acquisition"]["data_sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()
    parquet.write_bytes(parquet.read_bytes() + b"altered")
    assert main(args) == 2


@pytest.mark.parametrize(("kind", "doppler"), [("gnss", True), ("gnss", False), ("lps", False)])
@pytest.mark.parametrize("derivative", ["three_point", "five_point"])
@pytest.mark.parametrize("position_filter", ["none", "boxcar5", "binomial5"])
def test_position_reports_retain_sourced_limits_across_native_methods(
    kind: str, doppler: bool, derivative: str, position_filter: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_registry(root / "registry")
    by_id = {claim.id: claim for claim in registry.claims}
    validation = by_id["DEV-005"]
    load = by_id["CX-003"]
    assert validation.sources == ["linke2018"]
    assert "linke2018" in registry.sources
    assert "positional differentiation and GPS Doppler speed" in validation.statement
    assert "sampling rate, processing and threshold definitions" in load.statement
    assert all(source in registry.sources for source in load.sources)
    notes = (root / "docs/gnss-lps-engine.md").read_text()
    assert "Registry claims DEV-005 (Linke et al.) and CX-003" in notes

    case, declaration = track_case(kind, hz=25, doppler=doppler)
    result = process(
        case.observation,
        declaration,
        ProcessingOptions(position_filter=position_filter, derivative=derivative),
    )
    assert result.limitations == (
        "Receiver position describes the antenna or tag point.",
        "LPS positions are supplied position solutions. Anchor visibility checks provide "
        "no ranging residuals, per-sample covariance, or validation of the anchor survey.",
        "Declared clock and latency bounds exclude variable latency and undocumented "
        "processing. Native status depends on producer metadata and sampled timestamps.",
        "Threshold distance uses whole sampled intervals. Acceleration and distance "
        "depend on sampling, noise, filter choice and boundary support.",
        "Sensitivity uses alternate offline smoothing and unfiltered stride-two "
        "decimation of the supplied record, with no independent lower-rate measurement.",
    )
    assert (result.kinematics.doppler_speed_m_s is not None) is doppler


@pytest.mark.parametrize("kind", ["gnss", "lps"])
def test_position_cli_preserves_the_operating_limitations(kind: str, tmp_path: Path) -> None:
    case, declaration = track_case(kind, hz=25)
    case.write(tmp_path)
    declared = tmp_path / "declaration.json"
    declared.write_text(declaration.model_dump_json())
    path = tmp_path / "result.json"
    assert (
        main(
            [
                "gnss-lps",
                "process",
                str(tmp_path / "observation"),
                "--declaration",
                str(declared),
                "--derivative",
                "five_point",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    result = process(
        io.read(tmp_path / "observation"), declaration, ProcessingOptions(derivative="five_point")
    )
    assert json.loads(path.read_text())["limitations"] == list(result.limitations)
