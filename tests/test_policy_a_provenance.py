"""RES-675 (Policy A): provenance chain and emitted limitations for engines that carry them.

M: processing a seeded, exported bundle records the source's hash/identity, the record equals
an independently derived hash of the bundle that was read back from disk, re-running gives
identical output, and a one-sample change changes the recorded digest.
O: every result of the engine carries non-empty, sourced limitation statements.
Engines that do not yet emit source identity or limitations are not covered here and stay
computational blockers in the ledger report.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from test_gnss_lps_acquisition import track_case
from test_lpt_acquisition import LINE, LPT, MOTION

from dynamisbench_soccer.engines import imu as imu_engine
from dynamisbench_soccer.engines import load_cell, radar_laser
from dynamisbench_soccer.engines.force_platform import process as force_process
from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec
from dynamisbench_soccer.engines.gnss_lps import process as track_process
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.engines.lpt import MethodSpec as LptMethod
from dynamisbench_soccer.engines.lpt import process as lpt_process
from dynamisbench_soccer.engines.radar_laser_acquisition import ChannelStage
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import cmj
from dynamisbench_soccer.reference.motion_models import IsometricRamp
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import SignalBundle


def _roundtrip(bundle: SignalBundle, tmp_path: Path) -> SignalBundle:
    """Write and re-read: the sidecar's data_sha256 is computed by the writer, not the engine."""
    io.write(bundle, tmp_path / "bundle")
    return io.read(tmp_path / "bundle")


def _file_hash(tmp_path: Path) -> str:
    """SHA-256 of the written parquet file, computed here and not by the writer or an engine."""
    parquet = next(p for p in io.paths_for(tmp_path / "bundle") if p.suffix == ".parquet")
    return hashlib.sha256(parquet.read_bytes()).hexdigest()


def _jsonable(report: object) -> str:
    return json.dumps(
        report,
        sort_keys=True,
        default=lambda o: dataclasses.asdict(o) if dataclasses.is_dataclass(o) else str(o),
    )


def _signal_channel(bundle: SignalBundle) -> str:
    return next(c.name for c in bundle.meta.channels if c.name != "time")


def _bump(bundle: SignalBundle, channel: str, delta: float = 1.0) -> SignalBundle:
    data = {k: v.copy() for k, v in bundle.data.items()}
    data[channel][len(data[channel]) // 2] += delta
    return SignalBundle(bundle.meta.model_copy(update={"data_sha256": None}), data)


def _force(bundle: SignalBundle, plates: int = 1):  # type: ignore[no-untyped-def]
    return force_process(
        bundle, plates=tuple(PlateSpec(plate=i + 1, assume_field_axes=True) for i in range(plates))
    )


@pytest.mark.parametrize(
    ("sub", "plates"), [("force_platform.single", 1), ("force_platform.dual", 2)]
)
def test_force_platform_results_record_source_hashes_and_reproduce_exactly(
    sub: str, plates: int, tmp_path: Path
) -> None:
    source = _roundtrip(export_case(sub, cmj(), hz=1000).observation, tmp_path)
    first = _force(source, plates).report()
    again = _force(source, plates).report()
    assert _jsonable(first) == _jsonable(again)
    spec = first["method_spec"]
    assert (
        spec["source_file_sha256"] == [source.meta.data_sha256]
        or len(spec["source_file_sha256"]) == plates
    )
    assert source.meta.data_sha256 in spec["source_file_sha256"]
    assert len(spec["input_digests"]) == plates and all(len(d) == 64 for d in spec["input_digests"])
    assert len(spec["qc_rules_sha256"]) == 64
    changed = _force(
        _roundtrip(_bump(source, "fz" if plates == 1 else "fz_left"), tmp_path / "x"), plates
    ).report()
    assert changed["method_spec"]["input_digests"] != spec["input_digests"]


@pytest.mark.parametrize(
    "sub", ["load_cell_isometric.load_cell", "load_cell_isometric.isometric_rig"]
)
def test_load_cell_results_record_input_digest_and_reproduce_exactly(
    sub: str, tmp_path: Path
) -> None:
    source = _roundtrip(export_case(sub, IsometricRamp(), hz=1000).observation, tmp_path)
    run = lambda b: load_cell.process(b, CellSpec(), load_cell.MethodSpec()).report()  # noqa: E731
    first, again = run(source), run(source)
    assert _jsonable(first) == _jsonable(again)
    spec = first["method_spec"]
    assert spec["source_file_sha256"] == source.meta.data_sha256 == _file_hash(tmp_path)
    assert len(spec["input_digest"]) == 64
    channel = _signal_channel(source)
    other = run(_roundtrip(_bump(source, channel), tmp_path / "x"))
    assert other["method_spec"]["input_digest"] != spec["input_digest"]


def test_lpt_results_record_bundle_identity_and_hash_and_reproduce_exactly(tmp_path: Path) -> None:
    source = _roundtrip(
        export_case("lpt_encoder.lpt", MOTION, test_id="cmj", hz=200).observation, tmp_path
    )
    run = lambda b: lpt_process(b, (LPT,), LINE, LptMethod(name="cubic_spline"))  # noqa: E731
    first, again = run(source), run(source)
    assert _jsonable(first) == _jsonable(again)
    assert first["bundle_id"] == source.meta.bundle_id
    assert first["data_sha256"] == source.meta.data_sha256 == _file_hash(tmp_path)
    changed = run(_roundtrip(_bump(source, _signal_channel(source)), tmp_path / "x"))
    assert changed["data_sha256"] != first["data_sha256"]


@pytest.mark.parametrize("kind", ["gnss", "lps"])
def test_track_results_record_acquisition_hashes_and_reproduce_exactly(
    kind: str, tmp_path: Path
) -> None:
    case, declaration = track_case(kind)
    source = _roundtrip(case.observation, tmp_path)  # sidecar hash set by the writer
    first = track_process(source, declaration)
    again = track_process(source, declaration)
    assert _jsonable(first) == _jsonable(again)
    acquisition = first.acquisition
    assert acquisition.data_sha256 is not None  # a vacuous None would hide a broken chain
    assert acquisition.data_sha256 == source.meta.data_sha256 == _file_hash(tmp_path)
    assert acquisition.provenance["created_by"]
    changed = _roundtrip(_bump(source, _signal_channel(source), 1e-9), tmp_path / "x")
    assert track_process(changed, declaration).acquisition.data_sha256 != acquisition.data_sha256


def test_radar_result_records_the_source_identity_and_reproduces_exactly() -> None:
    obs = export_case("radar_laser.radar", hz=100).observation
    options = radar_laser.ProcessingOptions(
        channel_stage=ChannelStage.NATIVE, correction_method="declared_fixed_angle"
    )
    first, again = radar_laser.process(obs, options), radar_laser.process(obs, options)
    assert _jsonable(first) == _jsonable(again) and first.source_id
    assert first.source_stage == "native" and first.start_method == "record_origin"


# ---- O: limitations are emitted with every result ------------------------------------------


def _limitation_texts(sub: str) -> list[str]:
    if sub == "radar":
        obs = export_case("radar_laser.radar", hz=100).observation
        opts = radar_laser.ProcessingOptions(
            channel_stage=ChannelStage.NATIVE, correction_method="declared_fixed_angle"
        )
        return list(radar_laser.process(obs, opts).limitations)
    if sub == "laser":
        obs = export_case("radar_laser.laser", hz=100).observation
        opts = radar_laser.ProcessingOptions(channel_stage=ChannelStage.NATIVE)
        return list(radar_laser.process(obs, opts).limitations)
    if sub in {"gnss", "lps"}:
        case, declaration = track_case(sub)
        return list(track_process(case.observation, declaration).limitations)
    if sub == "imu":
        obs = export_case("imu.inertial", hz=500).observation
        spec = imu_engine.ImuSpec(initial_orientation=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        return list(imu_engine.process(obs, spec).limitations)
    if sub in {"load_cell", "isometric_rig"}:
        name = f"load_cell_isometric.{sub}"
        obs = export_case(name, IsometricRamp(), hz=1000).observation
        return list(load_cell.process(obs, CellSpec(), load_cell.MethodSpec()).limitations)
    if sub == "lpt":
        obs = export_case("lpt_encoder.lpt", MOTION, test_id="cmj", hz=200).observation
        return list(lpt_process(obs, (LPT,), LINE, LptMethod(name="cubic_spline"))["limitations"])
    raise AssertionError(sub)


@pytest.mark.parametrize(
    "sub", ["radar", "laser", "gnss", "lps", "imu", "load_cell", "isometric_rig", "lpt"]
)
def test_every_result_carries_substantive_limitation_statements(sub: str) -> None:
    texts = _limitation_texts(sub)
    assert len(texts) >= 2 and all(len(t.split()) >= 3 for t in texts)
    assert len(set(texts)) == len(texts)
    joined = " ".join(texts).lower()
    assert any(word in joined for word in ("not ", "only", "assum", "declared", "conditional"))
    assert "interchangeable" not in joined.replace("not interchangeable", "")
