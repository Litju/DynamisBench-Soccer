"""Encoder-specific observation admission and counter reconstruction (RES-695)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.lpt import process
from dynamisbench_soccer.engines.lpt_acquisition import CableSpec, GeometrySpec, acquire
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import Phase, Trajectory
from dynamisbench_soccer.signals import io
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


@pytest.mark.parametrize(
    ("direction", "counts", "zero"),
    [
        (1, [252, 254, 0, 2, 4], 252),
        (-1, [4, 2, 0, 254, 252], 4),
    ],
)
def test_encoder_modular_count_rollover_preserves_signed_displacement_and_rate(
    direction: int, counts: list[int], zero: int
) -> None:
    observation = _count_observation(counts)
    declared = CableSpec(
        source="instrument",
        channel="counts",
        count_modulus=256,
        count_direction=direction,
        zero_count=zero,
        max_speed_m_s=4,
    )
    result = process(observation, (declared,), GEOMETRY)
    assert result["verdict"] == "processed"
    cable = result["cables"][0]
    assert cable["faults"] == {}
    np.testing.assert_allclose(
        cable["cable_displacement_m"], [0, 0.002, 0.004, 0.006, 0.008], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(result["motion"]["velocity_m_s"], [0.2] * 5, rtol=0, atol=1e-12)


def test_encoder_refuses_ambiguous_rollover_and_out_of_range_counts() -> None:
    observation = _count_observation([252, 254, 0, 2, 4])
    ambiguous = CableSpec(
        source="instrument", channel="counts", count_modulus=256, max_speed_m_s=13
    )
    with pytest.raises(ValueError, match="rollover is ambiguous"):
        acquire(observation, (ambiguous,), GEOMETRY)
    declared = CableSpec(source="instrument", channel="counts", count_modulus=256, max_speed_m_s=4)
    with pytest.raises(ValueError, match="encoder counts and zero"):
        acquire(_count_observation([252, 254, 256, 2, 4]), (declared,), GEOMETRY)


def test_encoder_rollover_does_not_mask_overspeed() -> None:
    observation = _count_observation([252, 254, 70, 72, 74])
    declared = CableSpec(source="instrument", channel="counts", count_modulus=256, max_speed_m_s=4)
    result = process(observation, (declared,), GEOMETRY)
    assert result["verdict"] == "reject"
    assert result["cables"][0]["faults"] == {"count_jump_or_overspeed": [2]}
    assert result["motion"] is None


def _count_observation(counts: list[int], *, hz: int = 100) -> SignalBundle:
    observation = export_case("lpt_encoder.encoder", MOTION, test_id="cmj", hz=hz).observation
    meta = observation.meta.model_dump()
    meta["n_rows"] = len(counts)
    meta["acquisition"]["sources"][0]["geometry"]["cable_angle_rad"] = 0
    return SignalBundle(
        BundleMeta.model_validate(meta),
        {"time": np.arange(len(counts)) / hz, "counts": np.array(counts, dtype=float)},
    )


@pytest.mark.parametrize("hz", [100, 128])
def test_encoder_count_allowance_survives_large_counter_origin(hz: int) -> None:
    modulus = 2**32
    observation = _count_observation([modulus - 1, 1, 3, 5, 7], hz=hz)
    # The declared speed gives one permitted count plus one quantisation count.
    declared = CableSpec(
        source="instrument",
        channel="counts",
        count_modulus=modulus,
        max_speed_m_s=hz / 1000,
    )
    cable = acquire(observation, (declared,), GEOMETRY).cables[0]
    assert cable.faults == {}


def test_encoder_cli_replays_verified_artifacts_and_detects_mutation(tmp_path: Path) -> None:
    stem, config, out = tmp_path / "observation", tmp_path / "config.json", tmp_path / "result.json"
    args = ["lpt-encoder", "process", str(stem), "--config", str(config), "--out", str(out)]
    config.write_text(
        json.dumps({"cables": [CABLE.model_dump()], "geometry": GEOMETRY.model_dump()})
    )

    artifacts = []
    for _ in range(2):
        case = export_case("lpt_encoder.encoder", MOTION, test_id="cmj", hz=100)
        parquet, sidecar = io.write(case.observation, stem)
        assert main(args) == 0
        raw = parquet.read_bytes()
        reported = json.loads(out.read_bytes())
        assert reported["data_sha256"] == hashlib.sha256(raw).hexdigest()
        assert reported["bundle_id"] == case.observation.meta.bundle_id
        assert reported["cables"][0]["spec"] == CABLE.model_dump()
        artifacts.append((raw, sidecar.read_bytes(), out.read_bytes()))
    assert artifacts[0] == artifacts[1]

    case.observation["counts"][7] += 1
    io.write(case.observation, stem)
    assert main(args) == 0
    new = json.loads(out.read_bytes())
    assert new["data_sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert new["data_sha256"] != reported["data_sha256"]
    parquet.write_bytes(parquet.read_bytes() + b"altered")
    assert main(args) == 2
    assert json.loads(out.read_bytes()) == new


@pytest.mark.parametrize(
    "protected_name", ["observation.parquet", "observation.json", "config.json"]
)
def test_encoder_cli_refuses_output_hard_link_to_input(
    protected_name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stem = tmp_path / "observation"
    config = tmp_path / "config.json"
    alias = tmp_path / "output.json"
    io.write(export_case("lpt_encoder.encoder", MOTION, test_id="cmj", hz=100).observation, stem)
    config.write_text(
        json.dumps({"cables": [CABLE.model_dump()], "geometry": GEOMETRY.model_dump()})
    )
    protected = tmp_path / protected_name
    original = protected.read_bytes()
    alias.hardlink_to(protected)
    assert (
        main(["lpt-encoder", "process", str(stem), "--config", str(config), "--out", str(alias)])
        == 2
    )
    assert "output path must not overwrite" in capsys.readouterr().err
    assert protected.read_bytes() == original
