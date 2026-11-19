import numpy as np
import pytest

from dynamisbench_soccer.engines.lpt_acquisition import CableSpec, GeometrySpec, acquire
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import Phase, Trajectory
from dynamisbench_soccer.signals.schema import (
    Acquisition,
    AcquisitionSource,
    Alignment,
    BundleMeta,
    ChannelSpec,
    Clock,
    Device,
    Geometry,
    SignalBundle,
)

MOTION = Trajectory(mass_kg=70, phases=(Phase("drive", 1, 2, True),), initial_velocity_m_s=0.4)
LINE = GeometrySpec(attachment_point="COM", vertical_line_xy_m=(0, 0))
LPT = CableSpec(source="instrument", channel="cable_length", max_speed_m_s=4)
COUNTS = CableSpec(source="instrument", channel="counts", max_speed_m_s=4)


def _modify(bundle, *, meta=None, data=None):
    return SignalBundle(
        BundleMeta.model_validate(bundle.meta.model_dump() if meta is None else meta),
        bundle.data if data is None else data,
    )


def test_fixed_vertical_line_using_res669_observation_and_separate_com_oracle():
    case = export_case("lpt_encoder.lpt", MOTION, test_id="cmj", hz=128)
    result = acquire(case.observation, (LPT,), LINE)
    assert result.geometry_method == "fixed_vertical_line_above_anchor"
    np.testing.assert_allclose(
        result.vertical_displacement_m, 0.4 * result.time_s + result.time_s**2, atol=2e-15
    )
    assert result.cables[0].limitations
    with pytest.raises(ValueError, match="excluding analytical"):
        acquire(case.truth, (LPT,), LINE)


def test_counts_quantization_and_missing_geometry_stay_in_cable_coordinates():
    case = export_case("lpt_encoder.encoder", MOTION, test_id="cmj")
    result = acquire(case.observation, (COUNTS,), GeometrySpec(attachment_point="COM"))
    assert result.geometry_method == "cable_only"
    assert result.vertical_displacement_m is None
    cable = result.cables[0]
    assert cable.resolution_m == 0.001
    z = 0.4 * result.time_s + result.time_s**2
    assert np.max(np.abs(cable.length_m - np.hypot(1, z))) <= 0.0005
    with pytest.raises(ValueError, match="differs"):
        acquire(case.observation, (COUNTS,), GeometrySpec(attachment_point="barbell"))


def test_native_irregular_timing_and_indexed_integrity_diagnostics():
    case = export_case("lpt_encoder.lpt", MOTION, test_id="cmj", hz=100)
    bundle = case.observation
    data = {name: values.copy() for name, values in bundle.data.items()}
    data["time"][1] += 0.001
    data["time"][50:] += 0.1
    data["cable_length"][30] = 4
    slack = np.zeros(bundle.meta.n_rows)
    slack[10] = 1
    meta = bundle.meta.model_dump()
    meta["channels"].append(dict(name="slack", quantity="dimensionless", unit="1"))
    meta["acquisition"]["sources"][0]["channels"].append("slack")
    data["slack"] = slack
    spec = LPT.model_copy(update={"slack_channel": "slack", "full_length_m": 4})
    result = acquire(_modify(bundle, meta=meta, data=data), (spec,), LINE)
    np.testing.assert_array_equal(result.time_s, data["time"])
    assert result.geometry_method == "integrity_failed"
    assert result.vertical_displacement_m is None
    assert result.cables[0].faults == {
        "dropout_gap": [50],
        "overspeed": [30],
        "retraction_overspeed": [31],
        "slack": [10],
        "full_extension": [30],
    }


def _dual(*, skew=0.0, uncertainty=0.0, bracketing=True):
    bundle = export_case("lpt_encoder.lpt", MOTION, test_id="cmj").observation
    t = bundle["time"]
    x, z = 0.1 + 0.2 * t, 1 + 0.4 * t + t**2
    left = np.hypot(x + 1, z)
    right = np.hypot(x - 1, z - 0.25)
    offset = 0.2
    sources = [
        AcquisitionSource(
            id=side,
            device=Device(device_class="lpt"),
            clock=side,
            channels=[f"time_{side}", f"length_{side}"],
            geometry=Geometry(
                attachment_point="barbell", cable_anchors_m=[anchor], unknown=["cable_angle_rad"]
            ),
            raw_format="synthetic test",
            calibration_unknown_reason="prescribed lengths in metres",
        )
        for side, anchor in (("left", (-1, 0, 0)), ("right", (1, 0, 0.25)))
    ]
    acq = Acquisition(
        sources=sources,
        clocks=[
            Clock(id=side, source="device", epoch="test trigger") for side in ("left", "right")
        ],
        reference_clock="left",
        alignments=[
            Alignment(
                clock="right",
                reference_clock="left",
                method="shared pulses",
                offset_s=offset,
                uncertainty_s=uncertainty,
                sync_events=[(-1, -1 + offset), (2, 2 + offset)] if bracketing else [(0, offset)],
            )
        ],
    )
    columns = [
        ChannelSpec(
            name=name,
            quantity="time" if name.startswith("time") else "distance",
            unit="s" if name.startswith("time") else "m",
        )
        for source in sources
        for name in source.channels
    ]
    meta = bundle.meta.model_dump()
    meta.update(channels=columns, acquisition=acq)
    return _modify(
        bundle,
        meta=meta,
        data=dict(time_left=t, length_left=left, time_right=t - offset + skew, length_right=right),
    )


DUAL = tuple(
    CableSpec(source=side, channel=f"length_{side}", max_speed_m_s=4) for side in ("left", "right")
)
BAR = GeometrySpec(attachment_point="barbell", max_sync_error_s=0.0005)


def test_dual_reconstruction_with_lateral_motion_and_offset_clocks():
    result = acquire(_dual(), DUAL, BAR)
    t = result.time_s
    np.testing.assert_allclose(result.vertical_displacement_m, 0.4 * t + t**2, atol=4e-15)
    np.testing.assert_allclose(result.cables[1].time_s, t - 0.2)
    assert result.geometry_method == "planar_dual_circle_above_baseline"


@pytest.mark.parametrize(
    "options,message",
    [
        ({"skew": 0.001}, "error budget"),
        ({"uncertainty": 0.001}, "error budget"),
        ({"bracketing": False}, "must bracket"),
    ],
)
def test_dual_rejects_unsynchronized_clocks(options, message):
    with pytest.raises(ValueError, match=message):
        acquire(_dual(**options), DUAL, BAR)


def test_source_and_reference_columns_cannot_be_used_as_acquisition_inputs():
    bundle = _dual()
    meta = bundle.meta.model_dump()
    meta["acquisition"]["sources"][1]["device"]["device_class"] = "manual_protocol"
    reference_mixed = _modify(bundle, meta=meta)
    with pytest.raises(ValueError, match="different acquisition source"):
        acquire(reference_mixed, (DUAL[0].model_copy(update={"channel": "length_right"}),), BAR)
    with pytest.raises(ValueError, match="expected an LPT or encoder"):
        acquire(reference_mixed, (DUAL[1],), BAR)
    meta = bundle.meta.model_dump()
    meta["provenance"]["transforms"].append("RES-669 stage=analytical_truth")
    with pytest.raises(ValueError, match="excluding analytical"):
        acquire(_modify(bundle, meta=meta), DUAL, BAR)


def test_signed_encoder_affine_calibration_and_count_jump():
    bundle = export_case("lpt_encoder.encoder", MOTION, test_id="cmj").observation
    spec = COUNTS.model_copy(
        update={
            "count_direction": -1,
            "zero_count": 1000,
            "initial_length_m": 3,
        }
    )
    raw = 4000 - bundle["counts"]
    calibrated = _modify(bundle, data={"time": bundle["time"], "counts": raw})
    geometry = GeometrySpec(attachment_point="COM")
    cable = acquire(calibrated, (spec,), geometry).cables[0]
    assert cable.absolute
    np.testing.assert_allclose(cable.length_m, bundle["counts"] * 0.001, atol=1e-15)

    raw = raw.copy()
    raw[12] -= 200
    jumped = _modify(bundle, data={"time": bundle["time"], "counts": raw})
    assert acquire(jumped, (spec,), geometry).cables[0].faults == {
        "count_jump_or_overspeed": [12],
        "retraction_overspeed": [13],
    }
