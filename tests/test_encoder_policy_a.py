"""Encoder-specific observation admission and counter reconstruction (RES-695)."""

import numpy as np
import pytest

from dynamisbench_soccer.engines.lpt_acquisition import CableSpec, GeometrySpec, acquire
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import Phase, Trajectory
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

CABLE = CableSpec(source="instrument", channel="counts", max_speed_m_s=4)
GEOMETRY = GeometrySpec(attachment_point="COM")
MOTION = Trajectory(mass_kg=70, phases=(Phase("drive", 1, 2, True),), initial_velocity_m_s=0.4)


def test_encoder_integrity_refuses_wrong_stage_and_calibration() -> None:
    case = export_case("lpt_encoder.encoder", MOTION, test_id="cmj", hz=100)
    observation = case.observation
    assert acquire(observation, (CABLE,), GEOMETRY).cables[0].faults == {}
    with pytest.raises(ValueError, match="excluding analytical"):
        acquire(case.truth, (CABLE,), GEOMETRY)

    meta = observation.meta.model_dump()
    records = meta["acquisition"]["sources"][0]["calibration"]
    records[0].update(input_unit="m", output_unit="1", gain=1, offset=0)
    wrong_units = SignalBundle(BundleMeta.model_validate(meta), observation.data)
    with pytest.raises(ValueError, match="raw encoder counts need identity calibration"):
        acquire(wrong_units, (CABLE,), GEOMETRY)


def test_encoder_duplicate_timestamp_refusal() -> None:
    observation = export_case("lpt_encoder.encoder", MOTION, test_id="cmj", hz=100).observation
    time = observation["time"].copy()
    time[2] = time[1]
    with pytest.raises(ValueError, match="strictly increasing timestamps"):
        bundle = SignalBundle(observation.meta, {**observation.data, "time": time})
        acquire(bundle, (CABLE,), GEOMETRY)


def test_encoder_fractional_count_refusal() -> None:
    observation = export_case("lpt_encoder.encoder", MOTION, test_id="cmj", hz=100).observation
    counts = np.asarray(observation["counts"]) + 0.5
    with pytest.raises(ValueError, match="raw counts must be integers"):
        acquire(
            SignalBundle(observation.meta, {**observation.data, "counts": counts}),
            (CABLE,),
            GEOMETRY,
        )
