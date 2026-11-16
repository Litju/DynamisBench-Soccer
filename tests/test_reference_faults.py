"""Corruption effects, negative controls, stage isolation and Linux byte determinism."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.reference.exporters import (
    CORRUPTED_STAGE,
    DEVICES,
    OBSERVATION_STAGE,
    content_hash,
    export_case,
)
from dynamisbench_soccer.reference.faults import (
    ContinuousFault,
    ContinuousKind,
    EventFault,
    EventKind,
    InertialFault,
    InertialKind,
    PositionFault,
    PositionKind,
    ProtocolFault,
    ProtocolKind,
    inject,
)
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import BundleMeta


def test_continuous_fault_effects_and_decimated_sample_rate() -> None:
    clean = export_case("radar_laser.radar").observation
    x = clean["velocity"]
    offset = inject(clean, ContinuousFault(ContinuousKind.OFFSET, ("velocity",), magnitude=2))
    np.testing.assert_allclose(offset["velocity"] - x, 2)
    drift = inject(clean, ContinuousFault(ContinuousKind.DRIFT, ("velocity",), magnitude=0.5))
    np.testing.assert_allclose(drift["velocity"] - x, 0.5 * (clean["time"] - clean["time"][0]))
    clipped = inject(clean, ContinuousFault(ContinuousKind.CLIPPING, ("velocity",), magnitude=2))
    np.testing.assert_array_equal(clipped["velocity"], np.clip(x, -2, 2))
    noisy = inject(
        clean, ContinuousFault(ContinuousKind.NOISE, ("velocity",), magnitude=0.1, seed=42)
    )
    np.testing.assert_array_equal(
        noisy["velocity"] - x, (x + np.random.default_rng(42).normal(0, 0.1, len(x))) - x
    )
    sampled = inject(clean, ContinuousFault(ContinuousKind.RESAMPLE, ("velocity",), stride=3))
    np.testing.assert_array_equal(sampled["velocity"], x[::3])
    assert sampled.meta.sampling.nominal_hz == pytest.approx(clean.meta.sampling.nominal_hz / 3)
    dropped = inject(clean, ContinuousFault(ContinuousKind.DROP, ("velocity",), stride=3))
    expected = np.delete(x, np.arange(1, len(x), 3))
    np.testing.assert_array_equal(dropped["velocity"], expected)
    assert dropped.meta.sampling.irregular


@pytest.mark.parametrize(
    "subfamily", ["timing_contact.gates", "timing_contact.contact", "lpt_encoder.encoder"]
)
@pytest.mark.parametrize("kind", list(EventKind))
def test_discrete_faults_have_exact_row_or_timestamp_effect(
    subfamily: str, kind: EventKind
) -> None:
    clean = export_case(subfamily).observation
    fault = EventFault(kind, index=1, latency_s=0.25)
    bad = inject(clean, fault)
    time = "t_s" if "t_s" in clean.data else "time"
    if kind is EventKind.MISSING:
        for name in clean.data:
            np.testing.assert_array_equal(bad[name], np.delete(clean[name], 1))
    elif kind is EventKind.DUPLICATE:
        assert bad.meta.n_rows == clean.meta.n_rows + 1
        assert bad[time][1] == bad[time][2]
    elif kind is EventKind.OUT_OF_ORDER:
        assert bad[time][1] > bad[time][2]
    else:
        assert bad[time][1] == pytest.approx(clean[time][1] + 0.25)
        for name in clean.data:
            if name != time:
                np.testing.assert_array_equal(bad[name], clean[name])


@pytest.mark.parametrize("subfamily", ["gnss_lps.lps", "gnss_lps.gnss"])
def test_position_faults_drop_jump_in_metres_and_interpolate_native_data(subfamily: str) -> None:
    from dynamisbench_soccer.reference.coordinates import LocalFrame

    clean = export_case(subfamily, hz=20).observation
    jumped = inject(clean, PositionFault(PositionKind.JUMP, index=5, count=2, shift_m=(2, -1, 0.5)))
    if subfamily.endswith("lps"):
        for name, shift in zip(
            ("position_x", "position_y", "position_z"), (2, -1, 0.5), strict=True
        ):
            np.testing.assert_allclose(jumped[name][5:7] - clean[name][5:7], shift)
    else:
        for i in (5, 6):
            frame = LocalFrame(
                float(clean["latitude"][i]), float(clean["longitude"][i]), float(clean["height"][i])
            )
            np.testing.assert_allclose(
                frame.from_geodetic(
                    float(jumped["latitude"][i]),
                    float(jumped["longitude"][i]),
                    float(jumped["height"][i]),
                ),
                [2, -1, 0.5],
                atol=3e-9,
            )
    dropped = inject(clean, PositionFault(PositionKind.DROPOUT, index=5, count=2))
    assert dropped.meta.n_rows == clean.meta.n_rows - 2
    assert dropped.meta.sampling.irregular
    interpolated = inject(clean, PositionFault(PositionKind.INTERPOLATED, stride=5))
    knots = np.unique(np.r_[np.arange(0, clean.meta.n_rows, 5), clean.meta.n_rows - 1])
    for name in clean.data:
        if name == "time":
            np.testing.assert_array_equal(interpolated[name], clean[name])
        else:
            np.testing.assert_array_equal(
                interpolated[name],
                np.interp(clean["time"], clean["time"][knots], clean[name][knots]),
            )
    assert any(not np.array_equal(interpolated[name], clean[name]) for name in clean.data)


@pytest.mark.parametrize("quantity", ["specific_force", "angular_velocity"])
def test_inertial_faults_use_declared_sensor_axes(quantity: str) -> None:
    clean = export_case("imu.inertial").observation
    names = [f"{quantity}_{a}" for a in "xyz"]
    original = np.column_stack([clean[name] for name in names])
    bias = inject(clean, InertialFault(InertialKind.BIAS, quantity=quantity, bias=(1, -2, 3)))
    np.testing.assert_allclose(
        np.column_stack([bias[n] for n in names]) - original,
        np.tile([1, -2, 3], (len(original), 1)),
    )
    rotation = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    misaligned = inject(
        clean, InertialFault(InertialKind.MISALIGNMENT, quantity=quantity, rotation=rotation)
    )
    np.testing.assert_allclose(
        np.column_stack([misaligned[n] for n in names]), original @ np.array(rotation).T
    )
    saturated = inject(clean, InertialFault(InertialKind.SATURATION, quantity=quantity, limit=0.1))
    np.testing.assert_array_equal(
        np.column_stack([saturated[n] for n in names]), np.clip(original, -0.1, 0.1)
    )


def test_protocol_faults_are_observable_without_mutating_true_force() -> None:
    pair = export_case("force_platform.dual")
    clean = pair.observation
    quiet = inject(clean, ProtocolFault(ProtocolKind.QUIET_MOTION, window_s=0.8, magnitude_n=50))
    first = clean["time"] < 0.8
    assert np.std((quiet["fz_left"] + quiet["fz_right"])[first]) > 30
    zeros = np.flatnonzero(clean["fz_left"] == 0)
    i = int(zeros[3])
    contact = inject(clean, ProtocolFault(ProtocolKind.DOUBLE_CONTACT, index=i, magnitude_n=100))
    assert contact["fz_left"][i] + contact["fz_right"][i] == 100
    step = inject(clean, ProtocolFault(ProtocolKind.STEP_OFF, index=10))
    np.testing.assert_array_equal(step["fz_left"][10:], 0)
    swapped = inject(clean, ProtocolFault(ProtocolKind.SIDE_SWAP))
    np.testing.assert_array_equal(swapped["fz_left"], clean["fz_right"])
    np.testing.assert_array_equal(swapped["fz_right"], clean["fz_left"])
    np.testing.assert_array_equal(swapped.total_vertical_force(), clean.total_vertical_force())
    # A second side swap is a metamorphic involution on values, never on provenance.
    recovered = inject(swapped, ProtocolFault(ProtocolKind.SIDE_SWAP))
    for name in clean.data:
        np.testing.assert_array_equal(recovered[name], clean[name])
    assert content_hash(recovered) != content_hash(clean)
    # Confirm independent QC detects applicable observable faults; not a validity oracle.
    from dynamisbench_soccer.qc.spec import load, run, verdict

    rules = load(Path(__file__).resolve().parents[1] / "qc/rules.toml")
    assert verdict(run(quiet, rules)) == "reject"
    assert verdict(run(step, rules)) == "reject"


def test_contact_protocol_double_contact_and_step_off() -> None:
    clean = export_case("timing_contact.contact").observation
    double = inject(clean, ProtocolFault(ProtocolKind.DOUBLE_CONTACT, index=1, window_s=0.05))
    np.testing.assert_array_equal(double["contact"], [1, 0, 1, 0, 1])
    assert np.all(np.diff(double["t_s"]) > 0)
    step = inject(clean, ProtocolFault(ProtocolKind.STEP_OFF, index=2))
    np.testing.assert_array_equal(step["contact"], [1, 0, 0])


def fault_for(subfamily: str):  # type: ignore[no-untyped-def]
    if subfamily == "imu.inertial":
        return InertialFault(InertialKind.BIAS, bias=(0.1, 0, 0))
    if subfamily.startswith("gnss_lps"):
        return PositionFault(PositionKind.JUMP)
    if subfamily.startswith("timing_contact") or subfamily.endswith("encoder"):
        return EventFault(EventKind.DUPLICATE)
    clean = export_case(subfamily).observation
    channel = next(c.name for c in clean.meta.channels if c.quantity.value != "time")
    return ContinuousFault(ContinuousKind.NOISE, (channel,), magnitude=0.1, seed=5)


@pytest.mark.parametrize("subfamily", sorted(DEVICES))
def test_faults_cover_all_subfamilies_with_isolated_lineage_and_deterministic_bytes(
    subfamily: str, tmp_path: Path
) -> None:
    pair = export_case(subfamily, hz=25)
    truth_before, clean_before = content_hash(pair.truth), content_hash(pair.observation)
    fault = fault_for(subfamily)
    a, b = inject(pair.observation, fault), inject(pair.observation, fault)
    assert content_hash(a) == content_hash(b)
    assert content_hash(pair.truth) == truth_before
    assert content_hash(pair.observation) == clean_before
    assert a.meta.data_category is DataCategory.SYNTHETIC
    assert CORRUPTED_STAGE in a.meta.provenance.transforms
    assert OBSERVATION_STAGE not in a.meta.provenance.transforms
    assert a.meta.provenance.parents[-1].sha256 == clean_before
    assert a.meta.provenance.parents[-1].bundle_id == pair.observation.meta.bundle_id
    assert a.meta.data_sha256 is None
    assert BundleMeta.model_validate(a.meta.model_dump()) == a.meta
    assert any(not np.array_equal(a[name], pair.observation[name]) for name in a.data)
    p, j = io.write(a, tmp_path / "a")
    p2, j2 = io.write(b, tmp_path / "b")
    assert p.read_bytes() == p2.read_bytes()
    assert j.read_bytes() == j2.read_bytes()
    recovered = io.read(tmp_path / "a")
    for name in a.data:
        np.testing.assert_array_equal(recovered[name], a[name])


def test_faults_refuse_truth_empirical_vendor_and_wrong_paradigms() -> None:
    from dynamisbench_soccer.signals.schema import VendorValue

    pair = export_case("force_platform.single")
    fault = ContinuousFault(ContinuousKind.OFFSET, ("fz",), magnitude=1)
    with pytest.raises(ValueError, match="only RES-669"):
        inject(pair.truth, fault)
    pair.observation.meta.data_category = DataCategory.EMPIRICAL
    with pytest.raises(ValueError):
        inject(pair.observation, fault)
    pair.observation.meta.data_category = DataCategory.SYNTHETIC
    pair.observation.meta.vendor_reported = {"oracle": VendorValue(value=1, unit="N")}
    with pytest.raises(ValueError):
        inject(pair.observation, fault)
    with pytest.raises(ValueError):
        inject(export_case("imu.inertial").observation, fault)
    with pytest.raises(ValueError):
        inject(export_case("force_platform.single").observation, EventFault(EventKind.MISSING))
    with pytest.raises(ValueError):
        inject(export_case("force_platform.single").observation, PositionFault(PositionKind.JUMP))
    with pytest.raises(ValueError):
        inject(export_case("force_platform.single").observation, InertialFault(InertialKind.BIAS))


@pytest.mark.parametrize(
    "fault",
    [
        ContinuousFault(ContinuousKind.OFFSET, ("time",), magnitude=1),
        ContinuousFault(ContinuousKind.NOISE, ("velocity",), magnitude=-1),
        ContinuousFault(ContinuousKind.CLIPPING, ("velocity",), magnitude=float("nan")),
        ContinuousFault(ContinuousKind.NOISE, ("velocity",), magnitude=1, seed=-1),
        ContinuousFault(ContinuousKind.DROP, ("velocity",), stride=0),
    ],
)
def test_invalid_continuous_faults_reject(fault: ContinuousFault) -> None:
    with pytest.raises(ValueError):
        inject(export_case("radar_laser.radar").observation, fault)


def test_other_fault_parameter_negative_controls() -> None:
    gates = export_case("timing_contact.gates").observation
    for fault in (
        EventFault(EventKind.MISSING, index=100),
        EventFault(EventKind.OUT_OF_ORDER, index=3),
        EventFault(EventKind.LATENCY, latency_s=-1),
    ):
        with pytest.raises(ValueError):
            inject(gates, fault)
    position = export_case("gnss_lps.lps").observation
    with pytest.raises(ValueError):
        inject(position, PositionFault(PositionKind.DROPOUT, count=position.meta.n_rows))
    imu = export_case("imu.inertial").observation
    bad_rotation = InertialFault(
        InertialKind.MISALIGNMENT, rotation=((1, 0, 0), (0, 1, 0), (0, 0, -1))
    )
    with pytest.raises(ValueError):
        inject(imu, bad_rotation)
    clean = export_case("force_platform.single").observation
    with pytest.raises(ValueError):
        inject(clean, ProtocolFault(ProtocolKind.SIDE_SWAP))
    with pytest.raises(ValueError):
        inject(clean, ProtocolFault(ProtocolKind.DOUBLE_CONTACT, index=1))
    # Strings cannot silently select a different typed fault family.
    with pytest.raises(ValueError):
        inject(clean, replace(ContinuousFault(ContinuousKind.OFFSET, ("fz",)), kind="offset"))


@pytest.mark.parametrize("subfamily", ["gnss_lps.lps", "gnss_lps.gnss"])
def test_seeded_position_noise_uses_metres_not_degrees(subfamily: str) -> None:
    from dynamisbench_soccer.reference.coordinates import LocalFrame

    clean = export_case(subfamily, hz=10).observation
    spec = PositionFault(
        PositionKind.NOISE, index=0, count=clean.meta.n_rows, seed=17, noise_sd_m=0.03
    )
    a = inject(clean, spec)
    assert content_hash(a) == content_hash(inject(clean, spec))
    shifts = np.random.default_rng(17).normal(0, 0.03, (clean.meta.n_rows, 3))
    if subfamily.endswith("lps"):
        actual = np.column_stack(
            [a[f"position_{axis}"] - clean[f"position_{axis}"] for axis in "xyz"]
        )
    else:
        actual = np.array(
            [
                LocalFrame(
                    float(clean["latitude"][i]),
                    float(clean["longitude"][i]),
                    float(clean["height"][i]),
                ).from_geodetic(
                    float(a["latitude"][i]), float(a["longitude"][i]), float(a["height"][i])
                )
                for i in range(clean.meta.n_rows)
            ]
        )
    np.testing.assert_allclose(actual, shifts, atol=3e-9)


@pytest.mark.parametrize(
    "subfamily,fault",
    [
        ("radar_laser.radar", ContinuousFault(ContinuousKind.OFFSET, ("velocity",), magnitude=0)),
        (
            "radar_laser.radar",
            ContinuousFault(ContinuousKind.CLIPPING, ("velocity",), magnitude=1000),
        ),
        ("timing_contact.gates", EventFault(EventKind.LATENCY, latency_s=0)),
        ("gnss_lps.lps", PositionFault(PositionKind.JUMP, shift_m=(0, 0, 0))),
        ("imu.inertial", InertialFault(InertialKind.BIAS)),
        ("imu.inertial", InertialFault(InertialKind.MISALIGNMENT)),
        ("force_platform.single", ProtocolFault(ProtocolKind.QUIET_MOTION, window_s=0.0001)),
    ],
)
def test_no_effect_requests_cannot_become_corrupted_observations(subfamily: str, fault) -> None:  # type: ignore[no-untyped-def]
    clean = export_case(subfamily).observation
    before = content_hash(clean)
    with pytest.raises(ValueError, match="no sample effect"):
        inject(clean, fault)
    assert content_hash(clean) == before


def test_contact_step_off_requires_loaded_start() -> None:
    clean = export_case("timing_contact.contact").observation
    with pytest.raises(ValueError, match="loaded contact"):
        inject(clean, ProtocolFault(ProtocolKind.STEP_OFF, index=1))


def test_corrupted_metadata_is_isolated_from_clean_artifacts() -> None:
    pair = export_case("imu.inertial")
    before = (content_hash(pair.truth), content_hash(pair.observation))
    bad = inject(pair.observation, InertialFault(InertialKind.BIAS, bias=(1, 0, 0)))
    bad.meta.protocol.protocol_id = "changed-corrupted-protocol"
    bad.meta.acquisition.frames[0].definition = "changed-corrupted-frame"
    assert (content_hash(pair.truth), content_hash(pair.observation)) == before
