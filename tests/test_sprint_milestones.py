"""RES-673: sprint milestone integration from the exact declared start."""

from __future__ import annotations

import numpy as np
import pytest

from dynamisbench_soccer.assessment import sprint
from dynamisbench_soccer.assessment.sprint import StartSpec


def _const(v: float, hz: float = 50, duration: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0, duration, 1 / hz)
    return t, np.full_like(t, v)


@pytest.mark.parametrize("trigger", [0.013, 0.2, 0.41])
def test_milestones_use_the_exact_fractional_start(trigger: float) -> None:
    t, v = _const(5.0)
    got = sprint.milestones(t, v, [5, 10], StartSpec(kind="external_trigger", trigger_s=trigger))
    assert got["5_m"] == pytest.approx(1.0, abs=1e-12)
    assert got["10_m"] == pytest.approx(2.0, abs=1e-12)


def test_milestones_for_a_linear_ramp_are_exact_between_samples() -> None:
    t = np.arange(0, 5, 0.1)
    v = 2.0 * t  # a = 2: x = t^2, so the arrival at d is sqrt(d)
    got = sprint.milestones(t, v, [0.37, 2.5, 7.3], StartSpec())
    for d in (0.37, 2.5, 7.3):
        assert got[f"{d:g}_m"] == pytest.approx(np.sqrt(d), abs=1e-12)
    shifted = sprint.milestones(t, v, [3.3], StartSpec(kind="external_trigger", trigger_s=0.05))
    # distance from t=0.05: x = t^2 - 0.05^2
    assert shifted["3.3_m"] == pytest.approx(np.sqrt(3.3 + 0.05**2) - 0.05, abs=1e-12)


def test_stationary_tail_does_not_move_the_first_arrival() -> None:
    t = np.arange(0, 6, 0.1)
    v = np.where(t <= 2.0 + 1e-9, 5.0, 0.0)  # 5 m/s, linear stop over the next sample, then rest
    # 10.0 m at t = 2.0; the stop adds 0.25 m: total 10.25 m reached at t = 2.1 and never exceeded
    got = sprint.milestones(t, v, [4.3, 10.0, 10.1, 10.25], StartSpec())
    assert got["4.3_m"] == pytest.approx(4.3 / 5.0, abs=1e-9)
    assert got["10_m"] == pytest.approx(2.0, abs=1e-9)
    tau = (5 - np.sqrt(25 - 4 * 25 * 0.1)) / 50  # 5*tau - 25*tau^2 = 0.1
    assert got["10.1_m"] == pytest.approx(2.0 + tau, abs=1e-9)
    assert got["10.25_m"] == pytest.approx(
        2.1, abs=1e-6
    )  # sqrt-conditioned at v=0; first arrival, not the rest period
    beyond = sprint.milestones(t, v, [10.26, 50], StartSpec())
    assert beyond == {"10.26_m": None, "50_m": None}  # no extrapolation past observed support


def test_milestone_at_zero_and_at_the_observed_end() -> None:
    t, v = _const(4.0, duration=2.0)
    got = sprint.milestones(t, v, [0, 7.8], StartSpec())
    assert got["0_m"] == pytest.approx(0.0)
    assert got["7.8_m"] == pytest.approx(7.8 / 4.0, abs=1e-9)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t, v: (t, np.where(np.arange(len(v)) == 5, -1.0, v)),  # negative speed
        lambda t, v: (t, np.where(np.arange(len(v)) == 5, np.nan, v)),
        lambda t, v: (np.where(np.arange(len(t)) == 5, t[4], t), v),  # duplicate time
        lambda t, v: (t[::-1], v),  # decreasing time
        lambda t, v: (t[:2], v[:2]),
    ],
)
def test_milestones_refuse_malformed_traces(mutate) -> None:  # type: ignore[no-untyped-def]
    t, v = _const(5.0)
    bad_t, bad_v = mutate(t, v)
    with pytest.raises(ValueError):
        sprint.milestones(bad_t, bad_v, [5], StartSpec())
    assert sprint.milestones(t, v, [5], StartSpec())["5_m"] == pytest.approx(1.0)
