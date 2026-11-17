"""RES-720 gates 0-3: pinned source, rights gate, fail-closed acquisition, strict parsing."""

from __future__ import annotations

import io as stdio
from pathlib import Path

import numpy as np
import pytest
from _external_synth import (
    REF,
    csv_bytes,
    rights_payload,
    sprint,
    upstream,
    verified_payload,
)
from pydantic import ValidationError

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.data import store
from dynamisbench_soccer.data.manifest import Usage
from dynamisbench_soccer.data.store import IntegrityError
from dynamisbench_soccer.external import acquire, rights, shorts_laser
from dynamisbench_soccer.external.models import (
    FileRole,
    RightsRecord,
    RightsStatus,
    SignalStage,
    SourceLock,
    load_lock,
    load_rights,
)

ALL = frozenset(FileRole)

# --- the committed shorts-laser records --------------------------------------------------


def test_committed_lock_pins_exact_bytes(repo_root: Path) -> None:
    lock = load_lock("shorts-laser", repo_root / "data/external")
    assert lock.pinned_ref == shorts_laser.PINNED_REF
    data = lock.file(shorts_laser.DATA_PATH)
    assert data.role is FileRole.DATA
    assert (data.sha256, data.size_bytes) == (shorts_laser.PINNED_SHA256, 160957)
    assert data.git_blob_sha1 == "4423e32f76b7e42560ed242cdeb331d2924a0570"
    assert all(shorts_laser.PINNED_REF in f.url for f in lock.files)
    assert lock.file("data-raw/laser-gun-data.R").role is FileRole.TRANSFORM_SCRIPT


def test_committed_rights_block_canonical_admission(repo_root: Path) -> None:
    record = load_rights("shorts-laser", repo_root / "data/external")
    assert record.license_id == "MIT"
    assert record.measurement_data_rights is RightsStatus.PENDING
    assert record.canonical_admission is RightsStatus.PENDING
    assert record.participant_consent == "not_documented"
    assert record.minors_possible and not record.redistribution_allowed
    assert any("measurement-data rights" in b for b in record.blockers)
    assert any("consent" in b for b in record.blockers)


def test_no_manifest_admits_the_rights_pending_source(repo_root: Path) -> None:
    assert not (repo_root / "data/manifests/shorts-laser.json").exists()


def test_cli_validate_and_rights(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = str(repo_root / "data/external")
    assert main(["external", "validate", "--root", root]) == 0
    assert main(["external", "rights", "shorts-laser", "--root", root]) == 0
    out = capsys.readouterr().out
    assert "canonical admission pending" in out and "BLOCKER" in out


# --- record validation --------------------------------------------------------------------


def test_lock_rejects_unpinned_urls(tmp_path: Path) -> None:
    raw = upstream(tmp_path).model_dump(mode="json")
    raw["files"][0]["url"] = raw["files"][0]["url"].replace(REF, "master")
    with pytest.raises(ValidationError, match="commit-pinned"):
        SourceLock.model_validate(raw)


def test_lock_requires_license_and_relative_paths(tmp_path: Path) -> None:
    raw = upstream(tmp_path).model_dump(mode="json")
    only_data = {**raw, "files": [raw["files"][0]]}
    with pytest.raises(ValidationError, match="license"):
        SourceLock.model_validate(only_data)
    raw["files"][0]["path"] = "../escape.csv"
    with pytest.raises(ValidationError):
        SourceLock.model_validate(raw)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"canonical_admission": "verified", "blockers": []}, "measurement-data"),
        ({"blockers": []}, "blockers"),
        ({"redistribution_allowed": True}, "redistribution"),
        ({"canonical_admission": "refused"}, "refused"),
    ],
)
def test_rights_record_cannot_claim_unsupported_permission(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        RightsRecord.model_validate(rights_payload(**overrides))


def test_verified_admission_still_needs_consent_and_no_minor_redistribution() -> None:
    with pytest.raises(ValidationError, match="consent"):
        RightsRecord.model_validate(verified_payload(participant_consent="not_documented"))
    with pytest.raises(ValidationError, match="minors"):
        RightsRecord.model_validate(
            verified_payload(minors_possible=True, redistribution_allowed=True)
        )
    assert RightsRecord.model_validate(verified_payload(redistribution_allowed=True))


# --- acquisition ----------------------------------------------------------------------------


def test_local_acquisition_verifies_and_is_read_only(tmp_path: Path) -> None:
    lock = upstream(tmp_path / "up")
    files = acquire.acquire(lock, source_dir=tmp_path / "up", root=tmp_path / "q", roles=ALL)
    assert set(files) == {f.path for f in lock.files}
    for path in files.values():
        assert path.stat().st_mode & 0o222 == 0
    assert acquire.verify_quarantine(lock, tmp_path / "q") == []


def test_acquisition_fails_closed_on_tampered_bytes(tmp_path: Path) -> None:
    lock = upstream(tmp_path / "up")
    csv = tmp_path / "up" / "data-raw/laser-gun-data.csv"
    data = bytearray(csv.read_bytes())
    data[-5] = ord("9") if data[-5] != ord("9") else ord("8")
    csv.write_bytes(bytes(data))
    with pytest.raises(IntegrityError, match="sha256"):
        acquire.acquire(lock, source_dir=tmp_path / "up", root=tmp_path / "q", roles=ALL)
    assert not (tmp_path / "q" / lock.source_id / REF / "data-raw/laser-gun-data.csv").exists()


def test_acquisition_refuses_symlinks(tmp_path: Path) -> None:
    lock = upstream(tmp_path / "up")
    csv = tmp_path / "up" / "data-raw/laser-gun-data.csv"
    real = tmp_path / "elsewhere.csv"
    real.write_bytes(csv.read_bytes())
    csv.unlink()
    csv.symlink_to(real)
    with pytest.raises(IntegrityError, match="regular file"):
        acquire.acquire(lock, source_dir=tmp_path / "up", root=tmp_path / "q", roles=ALL)


def test_download_is_bounded_and_verified(tmp_path: Path) -> None:
    lock = upstream(tmp_path / "up")
    served: dict[str, bytes] = {f.url: (tmp_path / "up" / f.path).read_bytes() for f in lock.files}
    requested: list[int] = []

    class Resp(stdio.BytesIO):
        def read(self, n: int = -1, /) -> bytes:
            requested.append(n)
            return super().read(n)

    files = acquire.acquire(
        lock, root=tmp_path / "q", opener=lambda u, _t: Resp(served[u]), roles=ALL
    )
    assert len(files) == 2 and requested == [f.size_bytes + 1 for f in lock.files]
    oversized = {u: b + b"x" for u, b in served.items()}
    with pytest.raises(IntegrityError, match="size"):
        acquire.acquire(
            lock, root=tmp_path / "q2", opener=lambda u, _t: Resp(oversized[u]), roles=ALL
        )


def test_default_opener_refuses_non_pinned_hosts() -> None:
    with pytest.raises(IntegrityError, match="non-pinned"):
        acquire._urlopen("http://example.org/file.csv", 1.0)


# --- rights gate ----------------------------------------------------------------------------


def _check(tmp_path: Path, **overrides: object) -> rights.RightsCheck:
    lock = upstream(tmp_path / "up")
    files = acquire.acquire(lock, source_dir=tmp_path / "up", root=tmp_path / "q", roles=ALL)
    return rights.verify(RightsRecord.model_validate(rights_payload(**overrides)), lock, files)


def test_rights_gate_permits_quarantine_but_blocks_canonical(tmp_path: Path) -> None:
    check = _check(tmp_path)
    assert check.license_evidence_verified
    rights.require_local_inspection(check)
    with pytest.raises(rights.RightsBlocked, match="synthetic blocker"):
        rights.require_canonical_admission(check)


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"license_grant_phrase": "All rights reserved, no permission granted"}, "grant phrase"),
        ({"copyright_holder": "Someone Else"}, "copyright holder"),
        ({"license_files": ["DESCRIPTION"]}, "not a license file"),
        ({"pinned_ref": "f" * 40}, "different pinned commit"),
    ],
)
def test_license_evidence_must_match_pinned_text(
    tmp_path: Path, overrides: dict[str, object], problem: str
) -> None:
    check = _check(tmp_path, **overrides)
    assert not check.license_evidence_verified
    assert any(problem in p for p in check.problems)
    with pytest.raises(rights.RightsBlocked):
        rights.require_local_inspection(check)


def test_manifest_refused_while_pending(tmp_path: Path) -> None:
    lock = upstream(tmp_path / "up")
    record = RightsRecord.model_validate(rights_payload())
    check = _check(tmp_path)
    with pytest.raises(rights.RightsBlocked, match="canonical admission"):
        acquire.to_manifest(lock, record, check, title="t", citation="c")


def test_canonical_path_uses_existing_store_when_verified(tmp_path: Path) -> None:
    lock = upstream(tmp_path / "up")
    files = acquire.acquire(lock, source_dir=tmp_path / "up", root=tmp_path / "q", roles=ALL)
    record = RightsRecord.model_validate(verified_payload())
    check = rights.verify(record, lock, files)
    manifest = acquire.to_manifest(lock, record, check, title="Synthetic", citation="test")
    assert manifest.usage is Usage.CROSS_VALIDATION_ONLY and not manifest.redistribution_allowed
    qroot = tmp_path / "q" / lock.source_id / REF
    staged = acquire.admit_canonical(
        manifest, qroot, store_root=tmp_path / "store", staging=tmp_path / "stage"
    )
    assert (staged / "data-raw/laser-gun-data.csv").is_symlink()
    assert store.verify(manifest, tmp_path / "store") == []


# --- strict source parsing ---------------------------------------------------------------


def _write(tmp_path: Path, data: bytes) -> tuple[Path, str]:
    import hashlib

    path = tmp_path / "laser.csv"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def test_parse_round_trips_synthetic_source(tmp_path: Path) -> None:
    cols = sprint(n=500)
    path, sha = _write(tmp_path, csv_bytes(cols))
    table = shorts_laser.parse(path, sha)
    assert table.n_rows == 500 and list(table.columns) == list(shorts_laser.EXPECTED_HEADER)
    assert np.array_equal(table.columns["time"], cols["time"])
    prof = shorts_laser.profile(table)
    assert prof["uniform_grid"] and prof["n_duplicate_time"] == 0
    assert prof["grid_hz_from_median_dt"] == pytest.approx(1000.0)
    assert prof["distance_resolution_m"] == pytest.approx(0.001)


def test_parse_refuses_unpinned_bytes(tmp_path: Path) -> None:
    path, _ = _write(tmp_path, csv_bytes(sprint(n=50)))
    with pytest.raises(shorts_laser.SourceIntegrityError, match="not the pinned"):
        shorts_laser.parse(path)  # default expectation is the real pinned SHA-256


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b.replace(b"raw_velocity", b"velocity_raw", 1), "header"),
        (lambda b: b.replace(b"time,", b"time,extra,", 1), "header"),
        (lambda b: b"\xef\xbb\xbf" + b, "byte-order"),
        (lambda b: b.replace(b"\n-0.011,", b",1\n-0.011,", 1), "fields"),
        (lambda b: b.replace(b"\n-0.011,", b"\nNA,", 1), "not finite"),
        (lambda b: b.replace(b"\n-0.011,", b"\ninf,", 1), "not finite"),
        (lambda b: b.replace(b"\n-0.011,", b"\n,", 1), "not finite"),
        (lambda b: b.replace(b"\n", b"\r", 1), "carriage"),
        (lambda b: b.replace(b"0.000", "0.00µ".encode(), 1), "non-ASCII"),
    ],
)
def test_parse_strict_format(tmp_path: Path, mutate: object, message: str) -> None:
    assert callable(mutate)
    path, sha = _write(tmp_path, mutate(csv_bytes(sprint(n=50))))
    with pytest.raises(shorts_laser.SourceFormatError, match=message):
        shorts_laser.parse(path, sha)


def test_parse_refuses_oversized_file(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, b"x" * (shorts_laser.MAX_BYTES + 1))
    with pytest.raises(shorts_laser.SourceFormatError, match="limit"):
        shorts_laser.parse(path, sha)


def test_profile_reports_timestamp_defects(tmp_path: Path) -> None:
    cols = sprint(n=200)
    cols["time"] = cols["time"].copy()
    cols["time"][50] = cols["time"][49]  # duplicate
    cols["time"][120:] += 0.004  # gap
    path, sha = _write(tmp_path, csv_bytes(cols))
    prof = shorts_laser.profile(shorts_laser.parse(path, sha))
    assert prof["n_duplicate_time"] == 1 and prof["n_nonpositive_dt"] == 1
    assert not prof["uniform_grid"] and prof["dt_max_s"] == pytest.approx(0.005)


def test_column_stages_are_never_native(tmp_path: Path) -> None:
    for col in shorts_laser.SOURCE_COLUMNS:
        assert col.stage is SignalStage.MANUFACTURER_PROCESSED
        assert "raw" not in col.channel  # 'raw_velocity' is not native raw data
        assert col.description.startswith("[manufacturer_processed]")
    raw = next(c for c in shorts_laser.SOURCE_COLUMNS if c.source_column == "raw_velocity")
    assert raw.channel == "velocity_vendor_unsmoothed" and "NOT native" in raw.meaning
    derived = dict(shorts_laser.PACKAGE_DERIVED_NOT_INGESTED)
    assert {"raw_acceleration", "butter_acceleration"} <= set(derived)
