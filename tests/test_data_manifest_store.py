import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamisbench_soccer.data import paths, store
from dynamisbench_soccer.data.manifest import (
    DataCategory,
    DatasetManifest,
    FileEntry,
    Source,
    Usage,
    describe_directory,
    sha256_file,
)


def _make_dataset(root: Path) -> Path:
    src = root / "src"
    (src / "trials").mkdir(parents=True)
    (src / "trials" / "cmj_001.csv").write_text("t,fz\n0,800\n0.001,801\n")
    (src / "README.txt").write_text("synthetic test dataset\n")
    return src


def _manifest(src: Path, **overrides: object) -> DatasetManifest:
    fields: dict[str, object] = {
        "dataset_id": "toy-cmj",
        "version": "1.0.0",
        "title": "Toy CMJ",
        "data_category": DataCategory.SYNTHETIC,
        "usage": Usage.TASK_INPUT,
        "license": "CC-BY-4.0",
        "redistribution_allowed": True,
        "source": Source(kind="local", location=str(src)),
        "files": describe_directory(src),
    }
    fields.update(overrides)
    return DatasetManifest.model_validate(fields)


def test_describe_directory_is_sorted_and_hashed(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path)
    entries = describe_directory(src)
    assert [e.path for e in entries] == ["README.txt", "trials/cmj_001.csv"]
    assert entries[1].sha256 == sha256_file(src / "trials" / "cmj_001.csv")


def test_manifest_roundtrip_is_stable(tmp_path: Path) -> None:
    m = _manifest(_make_dataset(tmp_path))
    out = tmp_path / "m.json"
    m.save(out)
    again = DatasetManifest.load(out)
    assert again == m
    assert again.to_json() == out.read_text()
    assert json.loads(out.read_text())["schema_version"] == 1


@pytest.mark.parametrize("bad", ["/abs/path", "../escape", "a\\b", ""])
def test_file_entry_rejects_unsafe_paths(bad: str) -> None:
    with pytest.raises(ValidationError):
        FileEntry(path=bad, sha256="0" * 64, size_bytes=1)


def test_manifest_rejects_duplicates_and_unknown_fields(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path)
    files = describe_directory(src)
    with pytest.raises(ValidationError, match="duplicate"):
        _manifest(src, files=[files[0], files[0]])
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate({**_manifest(src).model_dump(), "surprise": 1})


def test_non_excluded_dataset_requires_files(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path)
    with pytest.raises(ValidationError, match="must list their files"):
        _manifest(src, files=[])
    assert _manifest(src, files=[], usage=Usage.EXCLUDED).files == []


def test_ingest_verify_stage(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path)
    m = _manifest(src)
    store.ingest_local(m, src)
    assert store.verify(m) == []

    staged = store.stage(m)
    assert staged.is_relative_to(paths.staging_root())
    linked = staged / "trials" / "cmj_001.csv"
    assert linked.is_symlink()
    assert linked.read_text().startswith("t,fz")
    # one physical copy: the staged file *is* the store blob
    assert linked.resolve() == store.blob_path(m.files[1].sha256)
    assert not os.access(linked.resolve(), os.W_OK)


def test_tampered_store_fails_closed(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path)
    m = _manifest(src)
    store.ingest_local(m, src)
    blob = store.blob_path(m.files[1].sha256)
    blob.chmod(0o644)
    blob.write_text("t,fz\n0,999\n0.001,801\n")  # same length, different bytes

    problems = store.verify(m)
    assert len(problems) == 1 and "sha256" in problems[0].reason
    with pytest.raises(store.IntegrityError):
        store.stage(m)


def test_ingest_rejects_source_that_does_not_match_manifest(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path)
    m = _manifest(src)
    (src / "README.txt").write_text("changed after hashing\n")
    with pytest.raises(store.IntegrityError):
        store.ingest_local(m, src)
    assert not store.blob_path(m.files[0].sha256).exists()


def test_missing_blob_without_url_cannot_be_fetched(tmp_path: Path) -> None:
    m = _manifest(_make_dataset(tmp_path))
    with pytest.raises(store.IntegrityError, match="no url"):
        store.fetch(m)


def test_fetch_from_file_url(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path)
    files = [e.model_copy(update={"url": (src / e.path).as_uri()}) for e in describe_directory(src)]
    m = _manifest(src, files=files)
    store.fetch(m)
    assert store.verify(m) == []


def test_committed_manifests_are_valid(repo_root: Path) -> None:
    manifests = sorted((repo_root / "data" / "manifests").glob("*.json"))
    for path in manifests:
        DatasetManifest.load(path)
