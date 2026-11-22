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
