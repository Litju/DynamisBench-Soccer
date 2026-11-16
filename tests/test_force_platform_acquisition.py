"""RES-670 production acquisition checks with independent matrix/geometry values."""

import numpy as np
import pytest
from pydantic import ValidationError

from dynamisbench_soccer.engines.force_platform_acquisition import (
    PlateSpec,
    acquire,
    cop,
    input_digest,
)
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.signals.schema import BundleMeta, ChannelSpec, Quantity, SignalBundle


def full_wrench(values: np.ndarray) -> SignalBundle:
    source = export_case("force_platform.single", hz=1000).observation
    raw = source.meta.model_dump()
    channels = [
        ChannelSpec(
            name=prefix + axis,
            quantity=Quantity.FORCE if prefix == "f" else Quantity.TORQUE,
            unit="N" if prefix == "f" else "N*m",
            plate=1,
            axis=axis,
            frame="field",
        )
        for prefix in ("f", "m")
        for axis in "xyz"
    ]
    raw.update(
        n_rows=len(values),
        channels=[
            ChannelSpec(name="time", quantity=Quantity.TIME, unit="s").model_dump(),
            *[c.model_dump() for c in channels],
        ],
    )
    raw["acquisition"]["sources"][0]["channels"] = ["time", *[c.name for c in channels]]
    raw["acquisition"]["sources"][0]["calibration"] = []
    raw["acquisition"]["sources"][0]["calibration_unknown_reason"] = "test known wrench"
    raw["calibration"] = []
    return SignalBundle(
        BundleMeta.model_validate(raw),
        {
            "time": np.arange(len(values), dtype=np.float64) / 1000,
            **{c.name: values[:, i].astype(np.float64) for i, c in enumerate(channels)},
        },
    )


def test_matrix_zero_rotation_and_moment_origin_transport() -> None:
    expected = np.tile([10.0, 20.0, 100.0, 4.0, -3.0, 1.0], (100, 1))
    matrix = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
    volts = expected @ np.linalg.inv(matrix).T + 0.2
    bundle = full_wrench(expected)
    raw = bundle.meta.model_dump()
    for c in raw["channels"][1:]:
        c.update(quantity="voltage", unit="V")
    data = {
        "time": bundle["time"],
        **{c["name"]: volts[:, i] for i, c in enumerate(raw["channels"][1:])},
    }
    bundle = SignalBundle(BundleMeta.model_validate(raw), data)
    spec = PlateSpec(
        plate=1,
        channels=tuple(data)[1:],
        input_units=("V",) * 6,
        output_components=("fx", "fy", "fz", "mx", "my", "mz"),
        matrix=tuple(map(tuple, matrix)),
        raw_offsets=(0.2,) * 6,
        rotation=((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        origin_m=(1, 2, 0),
    )
    result = acquire(bundle, (spec,))
    expected_force = np.array([-20, 10, 100])
    expected_moment = np.array([3, 4, 1]) + np.cross([1, 2, 0], expected_force)
    np.testing.assert_allclose(
        result.plates[0].wrench, np.tile([*expected_force, *expected_moment], (100, 1)), atol=1e-12
    )
    combined, _ = cop(result)
    np.testing.assert_allclose(
        combined, np.tile([-expected_moment[1] / 100, expected_moment[0] / 100, 0], (100, 1))
    )


def test_per_plate_unloaded_zeroing_does_not_mutate_input() -> None:
    pair = export_case("force_platform.dual", hz=1000)
    data = {n: v.copy() for n, v in pair.observation.data.items()}
    data["fz_left"] += 7
    data["fz_right"] -= 3
    b = SignalBundle(pair.observation.meta, data)
    before = input_digest(b)
    # Independently known fixture's flight contains an unloaded calibration segment.
    result = acquire(
        b,
        (
            PlateSpec(plate=1, zero_window_s=(1.9, 2.0)),
            PlateSpec(plate=2, zero_window_s=(1.9, 2.0)),
        ),
    )
    np.testing.assert_allclose(
        result.vertical_force_n, pair.observation.total_vertical_force(), atol=1e-12
    )
    assert result.plates[0].offset[2] == 7
    assert result.plates[1].offset[2] == -3
    assert input_digest(b) == before
    assert cop(result)[0] is None


def test_cop_is_missing_without_moments_and_masked_at_low_load() -> None:
    b = full_wrench(
        np.array([[0.0, 0.0, 100.0, 20.0, -10.0, 0.0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    )
    combined, per_plate = cop(acquire(b))
    np.testing.assert_allclose(combined[0], [0.1, 0.2, 0])
    assert np.isnan(combined[1:]).all()
    assert np.isnan(per_plate[1][1:]).all()
    assert cop(acquire(export_case("force_platform.single").observation))[0] is None


def test_dual_clock_mapping_and_explicit_overlap_resampling() -> None:
    one = export_case(
        "force_platform.single", hz=1000, clock=DeviceClock(0.2, 0), latency_s=0.01
    ).observation
    two = export_case(
        "force_platform.single", hz=600, clock=DeviceClock(-0.3, 50), latency_s=0.02
    ).observation
    raw = two.meta.model_dump()
    raw["channels"][1]["plate"] = 2
    two = SignalBundle(BundleMeta.model_validate(raw), {n: v.copy() for n, v in two.data.items()})
    specs = (PlateSpec(plate=1, latency_s=0.01), PlateSpec(plate=2, latency_s=0.02))
    with pytest.raises(ValueError, match="grids differ"):
        acquire((one, two), specs)
    result = acquire((one, two), specs, alignment="linear_overlap")
    assert result.resampled
    assert result.time_s[0] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(result.vertical_force_n[:500], 2 * 75 * 9.80665, atol=1e-10)
    assert (
        result.time_s[-1]
        <= min(one["time"][-1] + 0.2 - 0.01, two["time"][-1] * (1 + 50e-6) - 0.3 - 0.02) + 1e-12
    )


def test_legacy_dual_requires_an_explicit_shared_clock_assumption() -> None:
    original = export_case("force_platform.dual", hz=1000).observation
    raw = original.meta.model_dump()
    raw["acquisition"] = None
    raw["acquisition_unknown_reason"] = "legacy CSV"
    b = SignalBundle(BundleMeta.model_validate(raw), original.data)
    with pytest.raises(ValueError, match="declared synchronization"):
        acquire(
            b,
            (
                PlateSpec(plate=1, assume_field_axes=True),
                PlateSpec(plate=2, assume_field_axes=True),
            ),
        )
    assert acquire(
        b,
        (PlateSpec(plate=1, assume_field_axes=True), PlateSpec(plate=2, assume_field_axes=True)),
        assume_shared_clock=True,
    ).assumed_clock


@pytest.mark.parametrize("kind", ["duplicates", "gap", "rate", "unknown_epoch"])
def test_bad_timebase_rejects(kind: str) -> None:
    b = export_case("force_platform.dual", hz=1000).observation
    raw = b.meta.model_dump()
    data = {n: v.copy() for n, v in b.data.items()}
    if kind == "duplicates":
        data["time"][10] = data["time"][9]
    elif kind == "gap":
        data["time"][10:] += 0.01
    elif kind == "rate":
        raw["sampling"]["nominal_hz"] = 500
    else:
        raw["acquisition"] = None
        raw["acquisition_unknown_reason"] = "unmapped clocks"
    bad = SignalBundle(BundleMeta.model_validate(raw), data)
    with pytest.raises(ValueError):
        acquire(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(matrix=((0.0,),), channels=("fz",), input_units=("N",), output_components=("fz",)),
        dict(matrix=((1.0,),), channels=("fz",), input_units=("V",), output_components=("fz",)),
        dict(rotation=((1, 0, 0), (0, 1, 0), (0, 0, -1))),
        dict(zero_offsets={"fx": 1}),
    ],
)
def test_invalid_calibration_rejects(kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises((ValueError, ValidationError)):
        spec = PlateSpec(plate=1, **kwargs)
        acquire(export_case("force_platform.single").observation, (spec,))


def test_cop_uses_elevated_and_tilted_plate_surface_not_global_ground() -> None:
    b = full_wrench(np.tile([10.0, 20.0, 100.0, 0.0, 0.0, 0.0], (100, 1)))
    point, _ = cop(acquire(b, (PlateSpec(plate=1, origin_m=(0, 0, 0.5)),)))
    np.testing.assert_allclose(point, np.tile([0, 0, 0.5], (100, 1)), atol=1e-12)
    rotation = ((1, 0, 0), (0, 0.8, -0.6), (0, 0.6, 0.8))
    force = np.array([10.0, 20.0, 100.0])
    local = np.array([0.1, 0.2, 0.0])
    wrench = np.r_[force, np.cross(local, force)]
    b = full_wrench(np.tile(wrench, (100, 1)))
    origin = np.array([1.0, 2.0, 0.5])
    point, _ = cop(acquire(b, (PlateSpec(plate=1, rotation=rotation, origin_m=tuple(origin)),)))
    np.testing.assert_allclose(
        point, np.tile(origin + np.array(rotation) @ local, (100, 1)), atol=1e-12
    )


def test_unknown_reference_clock_cannot_certify_dual_synchronization() -> None:
    b = export_case("force_platform.dual", hz=1000).observation
    raw = b.meta.model_dump()
    raw["acquisition"]["clocks"][0].update(source="unknown", epoch="unknown")
    unknown = SignalBundle(BundleMeta.model_validate(raw), b.data)
    with pytest.raises(ValueError, match="declared synchronization"):
        acquire(unknown)
    assert acquire(unknown, assume_shared_clock=True).assumed_clock


def test_metadata_coordinate_frames_resolve_and_unknown_or_mixed_frames_reject() -> None:
    b = full_wrench(np.tile([0.0, 0.0, 100.0, 0.0, 0.0, 0.0], (100, 1)))
    raw = b.meta.model_dump()
    for channel in raw["channels"][1:]:
        channel["frame"] = "sensor"
    raw["acquisition"]["frames"] = [f for f in raw["acquisition"]["frames"] if f["id"] != "sensor"]
    raw["acquisition"]["frames"].append(
        dict(
            id="sensor",
            kind="sensor",
            definition="sensor axes",
            parent="field",
            rotation_to_parent=((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        )
    )
    mapped = SignalBundle(BundleMeta.model_validate(raw), b.data)
    result = acquire(mapped)
    np.testing.assert_allclose(
        result.plates[0].wrench[:, :3], np.tile([100, 0, 0], (100, 1)), atol=1e-12
    )
    raw["acquisition"]["frames"][-1]["rotation_to_parent"] = None
    with pytest.raises(ValueError, match="rotation is not known"):
        acquire(SignalBundle(BundleMeta.model_validate(raw), b.data))
    raw["channels"][1]["frame"] = "field"
    with pytest.raises(ValueError, match="inconsistent"):
        acquire(
            SignalBundle(BundleMeta.model_validate(raw), b.data),
            (PlateSpec(plate=1, rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1))),),
        )


@pytest.mark.parametrize("field", ["subject", "session", "test", "protocol", "category"])
def test_unrelated_acquisitions_cannot_be_summed(field: str) -> None:
    one = export_case("force_platform.single", hz=1000).observation
    raw = one.meta.model_dump()
    raw["channels"][1]["plate"] = 2
    if field == "subject":
        raw["subject"]["pseudonym"] = "DIFFERENT"
    elif field == "session":
        raw["session"]["session_id"] = "DIFFERENT"
    elif field == "test":
        raw["test_id"] = "sj"
    elif field == "protocol":
        raw["protocol"]["protocol_id"] = "different.protocol"
    else:
        raw["data_category"] = "empirical"
    other = SignalBundle(BundleMeta.model_validate(raw), one.data)
    with pytest.raises(ValueError, match="must match"):
        acquire((one, other))
