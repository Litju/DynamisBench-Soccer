"""Accepted force-platform reconstructions against independent vertical motion."""

from collections.abc import Callable

import numpy as np
import pytest

from dynamisbench_soccer.engines.force_platform import MethodSpec, process
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import Trajectory, cmj, dj, hops, sj
from dynamisbench_soccer.signals.schema import SignalBundle


def _nonflat_observation(family: str, motion: Trajectory, rate: int) -> SignalBundle:
    observation = export_case(family, motion, hz=rate).observation
    force = [name for name in observation.data if name.startswith("fz")]
    time = np.arange(observation.meta.n_rows) / rate
    # A bounded loaded-only ripple distinguishes a held load from a stuck ADC
    # while keeping exact zero force in flight. The unperturbed oracle is unchanged.
    ripple = 0.05 / len(force) * np.sin(2 * np.pi * 17 * time)
    data = dict(observation.data)
    for name in force:
        data[name] = np.where(data[name] > 0, data[name] + ripple, data[name])
    return SignalBundle(observation.meta, data)


@pytest.mark.parametrize("rate", [600, 1000, 2000])
@pytest.mark.parametrize("family", ["force_platform.single", "force_platform.dual"])
@pytest.mark.parametrize("factory", [cmj, sj, dj, hops])
def test_force_platform_numerical_reconstruction_on_accepted_trials(
    rate: int, family: str, factory: Callable[[], Trajectory]
) -> None:
    motion = factory()
    observation = _nonflat_observation(family, motion, rate)
    external = factory in (dj, hops)
    weighing = _nonflat_observation(family, cmj(), rate) if external else None
    result = process(
        observation,
        method=MethodSpec(
            integration_start="record_start" if external else "quiet_start",
            initial_velocity_m_s=motion.initial_velocity_m_s if external else None,
        ),
        weighing=weighing,
    )
    assert result.verdict in {"accept", "accept_with_flags"}
    assert result.report()["diagnostic_only"] is False
    states = [motion.state(float(t)) for t in result.time_s]
    assert np.max(np.abs(result.velocity_m_s - [s.velocity_m_s for s in states])) < 0.06
    assert np.max(np.abs(result.displacement_m - [s.displacement_m for s in states])) < 0.07


@pytest.mark.parametrize(
    ("family", "plate_budgets_n"),
    [
        ("force_platform.single", (2.0,)),
        ("force_platform.dual", (0.6, 1.4)),
    ],
)
@pytest.mark.parametrize(
    ("hz", "method_name", "norm_bound"),
    [
        (600, "trapezoid", 1.0),
        (1000, "simpson", 1.5),
    ],
)
def test_accepted_force_platform_uncertainty_matches_independent_corner_extrema(
    family: str, plate_budgets_n: tuple[float, ...], hz: int, method_name: str, norm_bound: float
) -> None:
    motion = cmj()
    observation = _nonflat_observation(family, motion, hz)
    gravity = 9.80665
    total_error_n, weight_error_n, velocity_error_m_s = sum(plate_budgets_n), 2.5, 0.02
    result = process(
        observation,
        method=MethodSpec(
            integration=method_name,
            calibration_force_bound_n=total_error_n,
            body_weight_bound_n=weight_error_n,
            initial_velocity_bound_m_s=velocity_error_m_s,
        ),
    )
    assert result.verdict != "reject"

    times = np.arange(observation.meta.n_rows) / hz
    body_weight_n = 75 * gravity + np.sin(2 * np.pi * 17 * times[:hz]).mean() * 0.05
    # The peak lies in the landing phase (13.5 m/s²), above the propulsion phase.
    landing = (times > 1.75 + 5.4 / gravity) & (times < 1.95 + 5.4 / gravity)
    peak_n = 75 * (gravity + 13.5) + np.max(0.05 * np.sin(2 * np.pi * 17 * times[landing]))
    # Acceleration error is affine in force for fixed perturbed weight and
    # monotonic in weight for fixed perturbed force. Its extrema lie at corners.
    acceleration_error = max(
        abs(gravity * ((force + error_f) / (body_weight_n + error_w) - force / body_weight_n))
        for force in (0, peak_n)
        for error_f in (-total_error_n, total_error_n)
        for error_w in (-weight_error_n, weight_error_n)
    )
    duration_s = (observation.meta.n_rows - 1) / hz
    expected_velocity = velocity_error_m_s + norm_bound * duration_s * acceleration_error
    expected_displacement = norm_bound * duration_s * expected_velocity

    uncertainty = result.uncertainty
    assert uncertainty["velocity_input_error_envelope_m_s"] == pytest.approx(
        expected_velocity, rel=0, abs=1e-12
    )
    assert uncertainty["displacement_input_error_envelope_m"] == pytest.approx(
        expected_displacement, rel=0, abs=1e-12
    )
    assert uncertainty["sample_interval_s"] == pytest.approx(1 / hz, rel=0, abs=1e-12)
    assert uncertainty["event_interval_resolution_s"] == pytest.approx(
        1 / hz + 2e-12, rel=0, abs=1e-14
    )
    assert uncertainty["complete_measurement_uncertainty"] is False
