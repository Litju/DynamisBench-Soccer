"""RES-705: QC must not silently convert irregular timestamps to nominal time."""

from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.qc import spec
from dynamisbench_soccer.qc.checks import EVALUATORS
from dynamisbench_soccer.signals import examples
from dynamisbench_soccer.signals.schema import ChannelSpec, Quantity, SignalBundle

RULES = spec.load(Path(__file__).resolve().parents[1] / "qc/rules.toml")


def timestamped(scale: float = 1) -> SignalBundle:
    b = examples.sprint_radar()
    times = b.time() * scale
    meta = b.meta.model_copy(
        update={
            "channels": [
                ChannelSpec(name="time", quantity=Quantity.TIME, unit="s"),
                *b.meta.channels,
            ]
        }
    )
    return SignalBundle(meta, {"time": times, **b.data})


def test_uniform_measured_time_drives_acceleration_and_rate_diagnostics() -> None:
    b = timestamped(2)
    params = RULES.by_id("QC-R02").params
    actual = EVALUATORS["QC-R02"](b, params, None)
    nominal = EVALUATORS["QC-R02"](examples.sprint_radar(), params, None)
    assert actual.values["max_accel"] == pytest.approx(nominal.values["max_accel"] / 2)
    assert EVALUATORS["QC-S05"](b, RULES.by_id("QC-S05").params, None).status is spec.Status.FAIL


@pytest.mark.parametrize("fault", ["gap", "duplicate", "reverse"])
def test_irregular_time_fails_timing_qc_without_hiding_timestamp_diagnostics(fault: str) -> None:
    b = timestamped()
    if fault == "gap":
        b.data["time"][10:] += 1
    elif fault == "duplicate":
        b.data["time"][10] = b.data["time"][9]
    else:
        b.data["time"][10] = -1
    result = EVALUATORS["QC-R02"](b, RULES.by_id("QC-R02").params, None)
    assert result.status is spec.Status.FAIL
    assert "uniform increasing" in result.detail
    assert EVALUATORS["QC-S04"](b, RULES.by_id("QC-S04").params, None).status is spec.Status.FAIL
    assert spec.verdict(spec.run(b, RULES)) == "reject"


def test_zero_force_denominators_are_explicit_failures() -> None:
    b = examples.cmj_dual_plate()
    for values in b.data.values():
        values[:] = 0
    with np.errstate(all="raise"):
        assert (
            EVALUATORS["QC-S02"](b, RULES.by_id("QC-S02").params, None).status is spec.Status.FAIL
        )
        h = examples.hip_squeeze()
        for values in h.data.values():
            values[:] = 0
        assert (
            EVALUATORS["QC-H01"](h, RULES.by_id("QC-H01").params, None).status is spec.Status.FAIL
        )


def test_squat_jump_zero_mean_weighing_window_rejects() -> None:
    from dynamisbench_soccer.signals.schema import Sampling

    b = examples.cmj_dual_plate()
    n = b.meta.n_rows
    force = np.full(n, 50.0, dtype=np.float64)
    force[:1000] = np.tile([100.0, -100.0], 500)
    force[1400] = 1000
    force[1600:1900] = 0
    meta = b.meta.model_copy(update={"test_id": "sj", "sampling": Sampling(nominal_hz=1000)})
    b = SignalBundle(meta, {"fz_left": force / 2, "fz_right": force / 2})
    outcome = EVALUATORS["QC-T08"](b, RULES.by_id("QC-T08").params, None)
    assert outcome.status is spec.Status.FAIL
    assert "positive" in outcome.detail


def test_measured_rate_used_without_timestamp_channel() -> None:
    b = examples.sprint_radar()
    assert b.meta.sampling is not None
    b.meta.sampling.measured_hz = b.meta.sampling.nominal_hz / 2
    actual = EVALUATORS["QC-R02"](b, RULES.by_id("QC-R02").params, None)
    expected = np.max(np.abs(np.diff(b["velocity"]))) * b.meta.sampling.measured_hz
    assert actual.values["max_accel"] == pytest.approx(expected)
