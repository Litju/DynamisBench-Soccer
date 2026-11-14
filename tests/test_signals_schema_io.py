import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dynamisbench_soccer.signals import examples, io
from dynamisbench_soccer.signals.schema import BundleMeta, ChannelSpec, Quantity, SignalBundle

# Pinned output hashes: any change in serialization (library upgrade, writer options)
# must be a deliberate, reviewed change to these values.
GOLDEN_PARQUET_SHA256 = {
    "cmj_dual_plate": "582ebc7b4ef23f5ed668f8adda01c9a2189ffeec3027dbd0d91bf9a842671ccb",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", sorted(examples.ALL))
def test_roundtrip_is_byte_identical(name: str, tmp_path: Path) -> None:
    bundle = examples.ALL[name]()
    p1, j1 = io.write(bundle, tmp_path / "a" / name)
    loaded = io.read(tmp_path / "a" / name)
    p2, j2 = io.write(loaded, tmp_path / "b" / name)
    assert p1.read_bytes() == p2.read_bytes()
    assert j1.read_bytes() == j2.read_bytes()
    for col, values in bundle.data.items():
        np.testing.assert_array_equal(values, loaded[col])
    assert loaded.meta == bundle.meta


@pytest.mark.parametrize("name", sorted(examples.ALL))
def test_repeated_writes_are_deterministic(name: str, tmp_path: Path) -> None:
    p1, _ = io.write(examples.ALL[name](), tmp_path / "x")
    h1 = _sha(p1)
    p2, _ = io.write(examples.ALL[name](), tmp_path / "y")
    assert h1 == _sha(p2)


def test_golden_hashes(tmp_path: Path) -> None:
    for name, expected in GOLDEN_PARQUET_SHA256.items():
        path, _ = io.write(examples.ALL[name](), tmp_path / name)
        assert _sha(path) == expected, name


def test_sidecar_is_sorted_json_with_data_hash(tmp_path: Path) -> None:
    p, j = io.write(examples.cmj_dual_plate(), tmp_path / "cmj")
    raw = json.loads(j.read_text())
    assert raw["schema_version"] == 1
    assert raw["data_sha256"] == _sha(p)
    assert j.read_text() == json.dumps(raw, indent=2, sort_keys=True) + "\n"


def test_tampered_parquet_is_rejected(tmp_path: Path) -> None:
    p, _ = io.write(examples.nhe(), tmp_path / "nhe")
    other, _ = io.write(examples.hip_squeeze(), tmp_path / "other")
    p.write_bytes(other.read_bytes())
    with pytest.raises(io.BundleIntegrityError):
        io.read(tmp_path / "nhe")


def test_iter_batches_streams_bounded_chunks(tmp_path: Path) -> None:
    bundle = examples.cmj_dual_plate(fs=2000.0)
    io.write(bundle, tmp_path / "cmj")
    batches = list(io.iter_batches(tmp_path / "cmj", columns=["fz_left"], batch_rows=1000))
    assert all(len(b["fz_left"]) <= 1000 for b in batches)
    np.testing.assert_array_equal(
        np.concatenate([b["fz_left"] for b in batches]), bundle["fz_left"]
    )


def test_future_schema_version_is_refused(tmp_path: Path) -> None:
    _, j = io.write(examples.lpt(), tmp_path / "lpt")
    raw = json.loads(j.read_text())
    raw["schema_version"] = 2
    j.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="newer"):
        io.read_meta(tmp_path / "lpt")


def test_non_canonical_unit_rejected() -> None:
    with pytest.raises(ValidationError, match="canonical"):
        ChannelSpec(name="fz", quantity=Quantity.FORCE, unit="kgf")


def _meta_dict() -> dict:  # type: ignore[type-arg]
    return examples.cmj_dual_plate().meta.model_dump(mode="json")


def test_multi_plate_requires_summed_force_policy() -> None:
    d = _meta_dict()
    d["geometry"]["summed_force_policy"] = None
    with pytest.raises(ValidationError, match="summed_force_policy"):
        BundleMeta.model_validate(d)


def test_force_plate_channels_need_plate_number() -> None:
    d = _meta_dict()
    d["channels"][0]["plate"] = None
    with pytest.raises(ValidationError, match="declare their plate"):
        BundleMeta.model_validate(d)


def test_unknown_test_and_duplicate_channels() -> None:
    d = _meta_dict()
    d["test_id"] = "bench_press"
    with pytest.raises(ValidationError, match="unknown test_id"):
        BundleMeta.model_validate(d)
    d = _meta_dict()
    d["channels"][1]["name"] = "fz_left"
    with pytest.raises(ValidationError, match="duplicate"):
        BundleMeta.model_validate(d)


def test_events_bundle_requires_time_and_distance() -> None:
    d = examples.sprint_gates().meta.model_dump(mode="json")
    d["channels"] = [c for c in d["channels"] if c["name"] != "t_s"]
    with pytest.raises(ValidationError, match="t_s and distance_m"):
        BundleMeta.model_validate(d)


def test_data_must_be_finite_float64_and_match_channels() -> None:
    b = examples.nhe()
    bad = dict(b.data)
    bad["force_left"] = bad["force_left"].copy()
    bad["force_left"][3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SignalBundle(b.meta, bad)
    with pytest.raises(ValueError, match="must match channels"):
        SignalBundle(b.meta, {"force_right": b["force_right"], "force_left": b["force_left"]})
    with pytest.raises(ValueError, match="float64"):
        SignalBundle(b.meta, {k: v.astype(np.float32) for k, v in b.data.items()})


def test_vendor_values_are_kept_apart() -> None:
    meta = examples.cmj_dual_plate().meta
    assert "jump_height" in meta.vendor_reported
    assert "jump_height" not in [c.name for c in meta.channels]


def test_time_axis_and_total_force() -> None:
    b = examples.cmj_dual_plate(fs=1000.0)
    t = b.time()
    assert t[1] - t[0] == pytest.approx(0.001)
    np.testing.assert_allclose(b.total_vertical_force(), b["fz_left"] + b["fz_right"])
