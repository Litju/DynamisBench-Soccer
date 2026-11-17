"""RES-721 gates 0-3 for a Zenodo source: pinned record, rights, archive, members."""

from __future__ import annotations

import io as stdio
import json
import os
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
from _zenodo_synth import (
    BASE,
    KEY,
    RECORD_ID,
    build_archive,
    lock,
    record,
    relicensed,
    rights,
)
from pydantic import ValidationError

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.data.manifest import DatasetManifest, Usage
from dynamisbench_soccer.data.store import IntegrityError
from dynamisbench_soccer.external import rights as rg
from dynamisbench_soccer.external import zenodo
from dynamisbench_soccer.external.models import RightsStatus, load_lock, load_rights
from dynamisbench_soccer.external.zenodo_models import (
    ZenodoLock,
    load_zenodo_lock,
    load_zenodo_rights,
    metadata_digest,
)


@pytest.fixture
def src(tmp_path: Path) -> dict[str, Any]:
    archive = tmp_path / "up" / KEY
    files = build_archive(archive)
    meta = record(archive)
    return {"archive": archive, "files": files, "meta": meta, "lock": lock(archive, meta, files)}


def _check(src: dict[str, Any], **overrides: Any) -> rg.RightsCheck:
    return rg.verify_zenodo(rights(**overrides), src["lock"], src["meta"])


# --- committed records ---------------------------------------------------------------------


def test_committed_zenodo_lock_pins_exact_bytes(repo_root: Path) -> None:
    lk = load_zenodo_lock("uwb-optitrack", repo_root / "data/external")
    assert (lk.record_id, lk.doi, lk.license_id) == (3452007, "10.5281/zenodo.3452007", "cc-by-4.0")
    (a,) = lk.archives
    assert a.md5 == "d5221f12a5f093eacc21db797c0ac13a" and a.size_bytes == 570029
    assert a.sha256 == "0e26251bff1ddd1f868205c481c63137c4729d480444e60769c3240ba3fc4f38"
    names = {m.name.rsplit("/", 1)[-1] for m in a.members}
    assert {"UWB_Walk_dataset.csv", "UWB_Trolley_dataset.csv", "README_data.md"} <= names
    assert len(a.excluded) == 6 and all(e.name.startswith("__MACOSX/") for e in a.excluded)


def test_committed_rights_admit_cross_validation_only(repo_root: Path) -> None:
    rec = load_zenodo_rights("uwb-optitrack", repo_root / "data/external")
    assert rec.canonical_admission is RightsStatus.VERIFIED and not rec.redistribution_allowed
    assert rec.license_id == "CC-BY-4.0" and rec.license_grant_phrase == "cc-by-4.0"
    m = DatasetManifest.load(repo_root / "data/manifests/uwb-optitrack.json")
    lk = load_zenodo_lock("uwb-optitrack", repo_root / "data/external")
    assert m.usage is Usage.CROSS_VALIDATION_ONLY and m.license == "CC-BY-4.0"
    assert [(f.path, f.sha256, f.size_bytes, f.url) for f in m.files] == [
        (a.key, a.sha256, a.size_bytes, a.url) for a in lk.archives
    ]


def test_res720_records_unchanged(repo_root: Path) -> None:
    root = repo_root / "data/external"
    assert load_rights("shorts-laser", root).canonical_admission is RightsStatus.PENDING
    assert load_lock("shorts-laser", root).acquisition_date is None
    assert not (repo_root / "data/manifests/shorts-laser.json").exists()


def test_cli_validates_both_lock_kinds(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["external", "validate", "--root", str(repo_root / "data/external")]) == 0
    out = capsys.readouterr().out
    assert "shorts-laser" in out and "uwb-optitrack @ zenodo-3452007" in out


# --- lock and record validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["archives"][0].update(url="https://example.org/x.zip"), "record-pinned"),
        (lambda r: r.update(doi="10.5281/zenodo.1"), "own Zenodo DOI"),
        (lambda r: r.update(license_id="cc-by-nc-4.0"), "accepted open license"),
        (
            lambda r: r["archives"][0]["excluded"].append(
                {"name": r["archives"][0]["members"][0]["name"], "reason": "dup test"}
            ),
            "both",
        ),
        (lambda r: r["archives"][0]["members"][0].update(name="../x.csv"), "relative"),
    ],
)
def test_lock_rejects_unpinned_or_unsafe(src: dict[str, Any], mutate: Any, message: str) -> None:
    raw = src["lock"].model_dump(mode="json")
    mutate(raw)
    with pytest.raises(ValidationError, match=message):
        ZenodoLock.model_validate(raw)


def test_rights_record_requires_metadata_evidence() -> None:
    with pytest.raises(ValidationError, match="not license files"):
        rights(license_files=["LICENSE"])
    with pytest.raises(ValidationError, match="SPDX id"):
        rights(license_id="MIT")


def test_metadata_digest_ignores_counters_only(src: dict[str, Any]) -> None:
    meta = dict(src["meta"], stats={"views": 12345})
    assert metadata_digest(meta) == src["lock"].metadata_digest
    assert metadata_digest(relicensed(meta, "cc0-1.0")) != src["lock"].metadata_digest


# --- rights verification (before any archive byte) -----------------------------------------


def test_rights_verified_from_pinned_metadata(src: dict[str, Any]) -> None:
    check = _check(src)
    assert check.license_evidence_verified and check.local_inspection_permitted
    assert check.canonical_admission is RightsStatus.VERIFIED and rg.issued(check)


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda m: relicensed(m, "cc-by-nc-4.0"), "differs from the pinned digest"),
        (
            lambda m: {**m, "files": [{**m["files"][0], "checksum": "md5:" + "0" * 32}]},
            "differs from the pinned digest",
        ),
        (lambda m: {**m, "id": RECORD_ID + 1}, "not 999001"),
        (lambda m: {"id": RECORD_ID}, "malformed"),
    ],
)
def test_relicensed_or_changed_record_fails_closed(
    src: dict[str, Any], change: Any, problem: str
) -> None:
    check = rg.verify_zenodo(rights(), src["lock"], change(src["meta"]))
    assert not check.license_evidence_verified and not check.local_inspection_permitted
    assert any(problem in p for p in check.problems)


def test_holder_must_be_a_creator(src: dict[str, Any]) -> None:
    check = _check(src, copyright_holder="Someone Else Ltd")
    assert any("not a creator" in p for p in check.problems)


# --- archive acquisition --------------------------------------------------------------------


def test_archive_acquired_only_with_permitting_check(src: dict[str, Any], tmp_path: Path) -> None:
    denied = _check(
        src, local_inspection_permitted=False, canonical_admission="pending", blockers=["synthetic"]
    )
    src["archive"].chmod(0o000)  # any read of the archive bytes would raise
    try:
        with pytest.raises(rg.RightsBlocked):
            zenodo.acquire_archive(
                src["lock"], KEY, denied, source_dir=src["archive"].parent, root=tmp_path / "q"
            )
    finally:
        src["archive"].chmod(0o644)
    assert not (tmp_path / "q").exists()


def test_github_check_cannot_authorise_zenodo_source(
    src: dict[str, Any], tmp_path: Path, repo_root: Path
) -> None:
    from _external_synth import rights_payload, upstream

    from dynamisbench_soccer.external import acquire
    from dynamisbench_soccer.external.models import RightsRecord

    gl = upstream(tmp_path / "gh", source_id="synthetic-uwb")
    files = acquire.acquire(gl, source_dir=tmp_path / "gh", root=tmp_path / "q0")
    gh_check = rg.verify(
        RightsRecord.model_validate(rights_payload(source_id="synthetic-uwb")), gl, files
    )
    assert rg.issued(gh_check) and gh_check.local_inspection_permitted
    with pytest.raises(rg.RightsBlocked, match="different source"):
        zenodo.acquire_archive(
            src["lock"], KEY, gh_check, source_dir=src["archive"].parent, root=tmp_path / "q"
        )


def test_archive_verified_and_read_only(src: dict[str, Any], tmp_path: Path) -> None:
    path = zenodo.acquire_archive(
        src["lock"], KEY, _check(src), source_dir=src["archive"].parent, root=tmp_path / "q"
    )
    assert path.stat().st_mode & 0o222 == 0
    assert zenodo.audit_archive(src["lock"].archive(KEY), path) == []


def test_tampered_or_oversized_download_fails(src: dict[str, Any], tmp_path: Path) -> None:
    data = src["archive"].read_bytes()
    with pytest.raises(IntegrityError, match="md5"):
        zenodo.acquire_archive(
            src["lock"],
            KEY,
            _check(src),
            root=tmp_path / "q1",
            opener=lambda u, t: stdio.BytesIO(data[:-1] + bytes([data[-1] ^ 0xFF])),
        )
    with pytest.raises(IntegrityError, match="exceeds"):
        zenodo.acquire_archive(
            src["lock"],
            KEY,
            _check(src),
            root=tmp_path / "q2",
            opener=lambda u, t: stdio.BytesIO(data + b"x"),
        )


def test_default_opener_refuses_other_hosts() -> None:
    with pytest.raises(IntegrityError, match="non-Zenodo"):
        zenodo._urlopen("https://example.org/records/1", 1.0)


# --- archive audit and member reads --------------------------------------------------------


def _with_extra(tmp_path: Path, info: zipfile.ZipInfo, data: bytes) -> tuple[Any, Path]:
    archive = tmp_path / "bad" / KEY
    files = build_archive(archive, extra=[(info, data)])
    meta = record(archive)
    lk = lock(archive, meta, files)  # the lock does not list the extra entry
    return lk, archive


def test_audit_flags_unlisted_traversal_symlink_and_bomb(tmp_path: Path) -> None:
    link = zipfile.ZipInfo(BASE + "link.csv")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    cases = [
        (zipfile.ZipInfo(BASE + "extra.csv"), b"1,2\n", "not pinned and not excluded"),
        (zipfile.ZipInfo("../evil.csv"), b"x", "unsafe path"),
        (link, b"/etc/passwd", "symbolic link"),
    ]
    for i, (info, data, problem) in enumerate(cases):
        lk, archive = _with_extra(tmp_path / str(i), info, data)
        assert any(problem in p for p in zenodo.audit_archive(lk.archives[0], archive)), problem
    bomb = zipfile.ZipInfo(BASE + "bomb.csv")
    bomb.compress_type = zipfile.ZIP_DEFLATED
    lk, archive = _with_extra(tmp_path / "b", bomb, b"0" * 2_000_000)
    assert any("compression ratio" in p for p in zenodo.audit_archive(lk.archives[0], archive))


def test_audit_flags_missing_member(src: dict[str, Any], tmp_path: Path) -> None:
    files = dict(src["files"])
    files.pop(BASE + "UWB_Trolley_dataset.csv")
    smaller = tmp_path / "small" / KEY
    build_archive(smaller, files)
    problems = zenodo.audit_archive(src["lock"].archive(KEY), smaller)
    assert any("absent from the archive" in p for p in problems)


def test_member_read_in_memory_and_verified(src: dict[str, Any], tmp_path: Path) -> None:
    check = _check(src)
    path = zenodo.acquire_archive(
        src["lock"], KEY, check, source_dir=src["archive"].parent, root=tmp_path / "q"
    )
    name = BASE + "UWB_Walk_dataset.csv"
    before = set(os.listdir(path.parent))
    data = zenodo.read_member(src["lock"], KEY, path, name, check)
    assert data == src["files"][name] and set(os.listdir(path.parent)) == before  # no extraction
    with pytest.raises(KeyError):
        zenodo.read_member(src["lock"], KEY, path, "__MACOSX/" + BASE + "._README_data.md", check)


def test_member_bytes_must_match_pin(src: dict[str, Any], tmp_path: Path) -> None:
    name = BASE + "UWB_Walk_dataset.csv"
    files = dict(src["files"])
    raw = bytearray(files[name])
    raw[-3] = ord("9") if raw[-3] != ord("9") else ord("8")  # same size, different bytes
    files[name] = bytes(raw)
    swapped = tmp_path / "swap" / KEY
    build_archive(swapped, files)
    with pytest.raises(IntegrityError, match="differ from the pinned member"):
        zenodo.read_member(src["lock"], KEY, swapped, name, _check(src))


def test_manifest_only_with_canonical_admission(src: dict[str, Any]) -> None:
    pending = _check(src, canonical_admission="pending", blockers=["synthetic"])
    with pytest.raises(rg.RightsBlocked, match="canonical admission"):
        zenodo.to_manifest(
            src["lock"],
            rights(),
            pending,
            title="t",
            citation="c",
            privacy_notes="p",
            handling_notes="h",
        )
    m = zenodo.to_manifest(
        src["lock"],
        rights(),
        _check(src),
        title="t",
        citation="c",
        privacy_notes="p",
        handling_notes="h",
    )
    assert m.usage is Usage.CROSS_VALIDATION_ONLY and m.license == "CC-BY-4.0"
    assert m.source.doi == f"10.5281/zenodo.{RECORD_ID}"


def test_cli_acquire_offline_with_saved_metadata(
    src: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from _zenodo_synth import dump

    root = tmp_path / "ext"
    dump(src["lock"], root / "synthetic-uwb" / "source-lock.json")
    dump(rights(), root / "synthetic-uwb" / "rights.json")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(src["meta"]))
    code = main(
        [
            "external",
            "acquire",
            "synthetic-uwb",
            "--root",
            str(root),
            "--source-dir",
            str(src["archive"].parent),
            "--metadata",
            str(meta),
        ]
    )
    assert code == 0 and "canonical admission verified" in capsys.readouterr().out
    meta.write_text(json.dumps(relicensed(src["meta"], "cc-by-nc-4.0")))
    assert (
        main(
            [
                "external",
                "acquire",
                "synthetic-uwb",
                "--root",
                str(root),
                "--source-dir",
                str(src["archive"].parent),
                "--metadata",
                str(meta),
            ]
        )
        == 1
    )
