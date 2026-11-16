"""RES-669: actual serialized instrument observations against independent formulas."""

import math
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.qualification.ledger import SUBFAMILIES
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.coordinates import LocalFrame
from dynamisbench_soccer.reference.exporters import (
    DEVICES,
    OBSERVATION_STAGE,
    TRUTH_STAGE,
    content_hash,
    export_case,
)
from dynamisbench_soccer.reference.kinematics import G, cmj, dj, hops, sj
from dynamisbench_soccer.reference.spatial import GeodeticTrack, RigidBody
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import BundleMeta


def test_exporter_coverage_is_exactly_all_fifteen_subfamilies() -> None:
    assert set(DEVICES) == SUBFAMILIES


@pytest.mark.parametrize("subfamily", sorted(DEVICES))
def test_every_export_is_v2_deterministic_separate_and_hash_linked(
    subfamily: str, tmp_path: Path
) -> None:
    a = export_case(subfamily, hz=50, clock=DeviceClock(0.12, 50), latency_s=0.01)
    b = export_case(subfamily, hz=50, clock=DeviceClock(0.12, 50), latency_s=0.01)
    assert a.truth.meta.data_category is DataCategory.ANALYTICAL
    assert a.observation.meta.data_category is DataCategory.SYNTHETIC
    assert TRUTH_STAGE in a.truth.meta.provenance.transforms
    assert OBSERVATION_STAGE in a.observation.meta.provenance.transforms
    assert a.observation.meta.provenance.parents[0].sha256 == content_hash(a.truth)
    for artifact in (a.truth, a.observation):
        assert artifact.meta.schema_version == 2
        assert not artifact.meta.vendor_reported
        assert BundleMeta.model_validate(artifact.meta.model_dump()) == artifact.meta
        assert artifact.meta.acquisition is not None
    assert a.observation.meta.calibration
    digest = content_hash(a.truth)
    a.write(tmp_path / "a")
    b.write(tmp_path / "b")
    assert content_hash(a.truth) == digest
    for stem in ("truth", "observation"):
        for extension in ("json", "parquet"):
            assert (tmp_path / "a" / f"{stem}.{extension}").read_bytes() == (
                tmp_path / "b" / f"{stem}.{extension}"
            ).read_bytes()
        original = getattr(a, stem)
        recovered = io.read(tmp_path / "a" / stem)
        assert recovered.meta == original.meta
        for name in original.data:
            np.testing.assert_array_equal(recovered[name], original[name])
        batches = list(io.iter_batches(tmp_path / "a" / stem, batch_rows=7))
        for name in original.data:
            np.testing.assert_array_equal(
                np.concatenate([b[name] for b in batches]), original[name]
            )


@pytest.mark.parametrize("subfamily", sorted(DEVICES))
def test_exported_values_obey_independent_device_relations(subfamily: str) -> None:
    pair = export_case(subfamily, hz=47)
    truth, obs = pair.truth, pair.observation
    if subfamily.startswith("force_platform"):
        force = sum(obs[c] for c in obs.data if c.startswith("fz"))
        np.testing.assert_allclose(force, 75 * (G + truth["acceleration"]), atol=1e-10)
    elif subfamily.startswith("lpt_encoder"):
        squared = (
            (truth["position_x"] + 1) ** 2 + truth["position_y"] ** 2 + truth["position_z"] ** 2
        )
        if subfamily.endswith("lpt"):
            np.testing.assert_allclose(obs["cable_length"] ** 2, squared, atol=1e-12)
        else:
            assert np.max(np.abs(obs["counts"] / 1000 - np.sqrt(squared))) <= 0.0005 + 1e-12
    elif subfamily.startswith("load_cell_isometric") or subfamily == "nordic_hip.hip":
        recovered = (obs["voltage"] - 0.2) / 0.001 if "voltage" in obs.data else obs["force"]
        np.testing.assert_allclose(recovered, truth["force"], atol=1e-10)
    elif subfamily == "nordic_hip.nordic":
        np.testing.assert_allclose(
            (obs["force_left"] + obs["force_right"]) * 0.4, truth["resistance"], atol=1e-12
        )
        np.testing.assert_allclose(
            truth["gravity_torque"] - truth["resistance"] + truth["actuator_torque"], 0, atol=1e-12
        )
    elif subfamily == "timing_contact.gates":
        np.testing.assert_allclose(truth["position"], obs["distance_m"] - 0.1, atol=1e-11)
    elif subfamily == "timing_contact.contact":
        np.testing.assert_array_equal(obs["contact"], [1, 0, 1])
        assert obs["t_s"][2] - obs["t_s"][1] == pytest.approx(cmj().flight_time_s())
    elif subfamily == "radar_laser.radar":
        np.testing.assert_allclose(obs["velocity"], math.cos(0.2) * truth["velocity"], atol=1e-12)
    elif subfamily == "radar_laser.laser":
        np.testing.assert_allclose(obs["distance"] - 1, truth["position"], atol=1e-12)
    elif subfamily == "gnss_lps.gnss":
        field = np.array(
            [
                LocalFrame().from_geodetic(lat, lon, h)
                for lat, lon, h in zip(
                    obs["latitude"], obs["longitude"], obs["height"], strict=True
                )
            ]
        )
        np.testing.assert_allclose(
            field, np.column_stack([truth[f"position_{a}"] for a in "xyz"]), atol=3e-9
        )
    elif subfamily == "gnss_lps.lps":
        for a in "xyz":
            np.testing.assert_array_equal(obs[f"position_{a}"], truth[f"position_{a}"])
    else:
        for i in range(obs.meta.n_rows):
            r = np.array([[truth[f"rotation_{j}{k}"][i] for k in range(3)] for j in range(3)])
            specific = np.array([obs[f"specific_force_{a}"][i] for a in "xyz"])
            np.testing.assert_allclose(
                r @ specific + [0, 0, -G],
                [truth[f"acceleration_{a}"][i] for a in "xyz"],
                atol=1e-12,
            )


def test_native_rate_multidevice_sync_reaches_same_physical_truth_without_resampling() -> None:
    rates = {"gnss_lps.gnss": 10.0, "gnss_lps.lps": 25.0, "imu.inertial": 100.0}
    clocks = [DeviceClock(0.1, 40), DeviceClock(-0.2, -30), DeviceClock(0.03, 70)]
    pairs = [
        export_case(sub, hz=rate, clock=clock, latency_s=0.025)
        for (sub, rate), clock in zip(rates.items(), clocks, strict=True)
    ]
    assert len({p.observation.meta.n_rows for p in pairs}) == 3
    for pair, rate, clock in zip(pairs, rates.values(), clocks, strict=True):
        obs = pair.observation
        np.testing.assert_allclose(np.diff(obs["time"]), 1 / rate, atol=1e-12)
        np.testing.assert_allclose(
            clock.reference_time(obs["time"]) - 0.025, pair.truth["time"], atol=1e-12
        )
        assert obs.meta.sampling.nominal_hz == rate
        alignment = obs.meta.acquisition.alignments[0]
        assert alignment.offset_s == clock.offset_s
        assert alignment.drift_ppm == clock.drift_ppm
        # Every separate device samples exactly the same scenario at its measurement times.
        from dynamisbench_soccer.reference.exporters import default_scenario

        model = default_scenario("imu.inertial")
        assert isinstance(model, RigidBody)
        for i in (0, obs.meta.n_rows // 2, -1):
            expected = model.sensor_state(float(pair.truth["time"][i]))[0]
            np.testing.assert_allclose(
                expected, [pair.truth[f"position_{a}"][i] for a in "xyz"], atol=1e-12
            )


@pytest.mark.parametrize("motion", [cmj(), sj(), dj(), hops(5)])
@pytest.mark.parametrize(
    "subfamily",
    [
        "force_platform.single",
        "force_platform.dual",
        "lpt_encoder.lpt",
        "lpt_encoder.encoder",
        "imu.inertial",
        "timing_contact.contact",
    ],
)
def test_existing_vertical_models_all_have_exporters(motion, subfamily: str) -> None:  # type: ignore[no-untyped-def]
    assert export_case(subfamily, motion, hz=100).observation.meta.n_rows >= 2


def test_prescribed_geodetic_track_exports_without_invented_velocity() -> None:
    pair = export_case("gnss_lps.gnss", GeodeticTrack(40, -3, 10, (0.00001, 0.00002, 0)))
    np.testing.assert_array_equal(pair.truth["latitude"], pair.observation["latitude"])
    assert not any("velocity" in name for name in pair.observation.data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subfamily": "unknown"},
        {"subfamily": "imu.inertial", "scenario": cmj(), "hz": 0},
        {"subfamily": "imu.inertial", "hz": float("nan")},
        {"subfamily": "force_platform.single", "scenario": GeodeticTrack(0, 0, 0)},
        {"subfamily": "gnss_lps.lps", "hz": 1e10},
        {"subfamily": "timing_contact.gates", "latency_s": -1},
    ],
)
def test_invalid_export_requests_reject(kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        export_case(**kwargs)


@pytest.mark.parametrize(
    "motion,assessment", [(cmj(), "cmj"), (sj(), "sj"), (dj(), "dj"), (hops(5), "rj_10_5")]
)
def test_export_metadata_preserves_named_assessment(motion, assessment: str) -> None:  # type: ignore[no-untyped-def]
    pair = export_case("force_platform.single", motion)
    assert pair.truth.meta.test_id == assessment
    assert pair.observation.meta.test_id == assessment
    assert assessment in pair.observation.meta.protocol.protocol_id


def test_tracking_context_and_custom_assessments_are_explicit() -> None:
    assert export_case("gnss_lps.gnss").observation.meta.test_id == "sprint_fv"
    with pytest.raises(ValueError, match="contradicts"):
        export_case("force_platform.single", sj(), test_id="cmj")


def test_contact_without_switches_is_explicit_negative_control() -> None:
    from dynamisbench_soccer.reference.kinematics import Phase, Trajectory

    static = Trajectory(75, 0, (Phase("contact", 1, 0, True),), assessment_id="cmj")
    with pytest.raises(ValueError, match="at least two"):
        export_case("timing_contact.contact", static)


def test_prescribed_jump_and_hip_protocol_conditions_are_exported() -> None:
    for motion in (cmj(), sj(), dj(drop_height_m=0.42), hops(10), hops(5)):
        pair = export_case("force_platform.single", motion)
        assert pair.observation.meta.protocol.arm_swing == "hands_on_hips"
        if motion.assessment_id == "dj":
            assert pair.observation.meta.protocol.drop_height_m == 0.42
        if motion.assessment_id == "rj_10_5":
            assert (
                f"prescribed_hops_{len(motion.flight_intervals())}"
                in pair.observation.meta.protocol.protocol_id
            )
    assert export_case("nordic_hip.hip").observation.meta.protocol.action == "adduction"


def test_common_scenario_digest_excludes_device_sampling_and_clock_configuration() -> None:
    pairs = [
        export_case(name, hz=hz, clock=DeviceClock(offset, drift))
        for name, hz, offset, drift in (
            ("gnss_lps.gnss", 10, 0.1, 20),
            ("gnss_lps.lps", 25, -0.2, -30),
            ("imu.inertial", 100, 0.03, 50),
        )
    ]
    digests = [
        next(
            tag
            for tag in pair.observation.meta.provenance.transforms
            if tag.startswith("scenario_sha256=")
        )
        for pair in pairs
    ]
    assert len(set(digests)) == 1
    assert len({pair.observation.meta.bundle_id for pair in pairs}) == 3


@pytest.mark.parametrize("subfamily", sorted(DEVICES))
def test_sampling_refinement_preserves_coincident_observations(subfamily: str) -> None:
    low = export_case(subfamily, hz=25).observation
    high = export_case(subfamily, hz=50).observation
    step = 1 if low.meta.kind.value == "events" else 2
    for name in low.data:
        if name == "counts":
            np.testing.assert_array_equal(low[name], high[name][::step])
        else:
            np.testing.assert_allclose(low[name], high[name][::step], rtol=1e-12, atol=1e-12)


def test_mass_scaling_is_a_force_metamorphism_not_a_jump_height_change() -> None:
    low = export_case("force_platform.single", cmj(50))
    high = export_case("force_platform.single", cmj(100))
    np.testing.assert_allclose(high.observation["fz"], 2 * low.observation["fz"], atol=1e-12)
    np.testing.assert_array_equal(high.truth["height"], low.truth["height"])
    np.testing.assert_array_equal(high.truth["velocity"], low.truth["velocity"])
