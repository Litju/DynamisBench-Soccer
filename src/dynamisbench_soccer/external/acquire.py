"""Fail-closed acquisition of pinned source bytes.

Two destinations exist and they never mix:

* **Quarantine** (``$DBS_CACHE_HOME/quarantine/<source>/<commit>/``) - a read-only,
  hash-verified local copy used for inspection and diagnostics while canonical
  admission is not (yet) permitted. It is not the dataset store and is never staged.
* **Canonical store** - reached only through :func:`admit_canonical`, which requires a
  rights record that permits admission, then uses the existing manifest-v1
  ``ingest_local``/``verify``/``stage`` path unchanged.

Every file is checked for exact size, SHA-256 and Git blob SHA-1 before it is
installed; downloads are size-bounded; nothing downloaded is executed.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import urllib.request
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Protocol

from dynamisbench_soccer.data import paths, store
from dynamisbench_soccer.data.manifest import (
    DataCategory,
    DatasetManifest,
    FileEntry,
    Source,
    Usage,
)
from dynamisbench_soccer.data.store import IntegrityError
from dynamisbench_soccer.external.models import (
    FileRole,
    PinnedFile,
    RightsRecord,
    SourceLock,
)
from dynamisbench_soccer.external.rights import (
    RightsBlocked,
    RightsCheck,
    require_canonical_admission,
    require_local_inspection,
)


class _Readable(Protocol):
    def read(self, n: int = -1, /) -> bytes: ...


Opener = Callable[[str, float], _Readable]

DATA_ROLES: frozenset[FileRole] = frozenset({FileRole.DATA})
EVIDENCE_ROLES: frozenset[FileRole] = frozenset(FileRole) - DATA_ROLES


def _urlopen(url: str, timeout: float) -> _Readable:
    if not url.startswith("https://raw.githubusercontent.com/"):
        raise IntegrityError(f"refusing non-pinned or non-HTTPS url {url!r}")
    response: _Readable = urllib.request.urlopen(url, timeout=timeout)
    return response


def quarantine_root() -> Path:
    return paths.cache_home() / "quarantine"


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def check_bytes(entry: PinnedFile, data: bytes) -> list[str]:
    problems = []
    if len(data) != entry.size_bytes:
        problems.append(f"size {len(data)} != pinned {entry.size_bytes}")
    sha = hashlib.sha256(data).hexdigest()
    if sha != entry.sha256:
        problems.append(f"sha256 {sha[:12]}... != pinned {entry.sha256[:12]}...")
    blob = git_blob_sha1(data)
    if blob != entry.git_blob_sha1:
        problems.append(f"git blob {blob[:12]}... != pinned {entry.git_blob_sha1[:12]}...")
    return problems


def _read_local(path: Path, limit: int) -> bytes:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise IntegrityError(f"{path}: must be a regular file, not a link or special file")
    if info.st_size > limit:
        raise IntegrityError(f"{path}: {info.st_size} B exceeds pinned size")
    return path.read_bytes()


def _install(entry: PinnedFile, data: bytes, dest: Path) -> None:
    problems = check_bytes(entry, data)
    if problems:
        raise IntegrityError(f"{entry.path}: " + "; ".join(problems))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".incoming-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        tmp.chmod(0o444)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def acquire(
    lock: SourceLock,
    *,
    source_dir: Path | None = None,
    root: Path | None = None,
    opener: Opener | None = None,
    timeout: float = 60.0,
    roles: Collection[FileRole] = EVIDENCE_ROLES,
    check: RightsCheck | None = None,
) -> dict[str, Path]:
    """Install the pinned files of ``roles`` into quarantine, verified.

    Returns ``path -> local file``. By default only **evidence** (license, documentation,
    transform scripts, acquisition evidence) is acquired. Measurement-data bytes are
    acquired only when a caller passes ``roles=DATA_ROLES``, which callers do only after
    the rights gate permits it (:func:`acquire_data`). A file outside ``roles`` is never
    downloaded, opened or hashed, including a copy already sitting in quarantine.

    With ``source_dir`` the bytes are read from a local copy laid out like the upstream
    repository; otherwise they are downloaded from the commit-pinned URLs.
    """
    if set(roles) & DATA_ROLES:
        if check is None:
            raise RightsBlocked("local quarantine inspection", ["no rights check supplied"])
        require_local_inspection(check, lock)
    target = (root or quarantine_root()) / lock.source_id / lock.pinned_ref
    out: dict[str, Path] = {}
    for entry in (f for f in lock.files if f.role in roles):
        dest = target / entry.path
        if dest.is_file() and not dest.is_symlink() and not check_bytes(entry, dest.read_bytes()):
            out[entry.path] = dest
            continue
        if source_dir is not None:
            data = _read_local(source_dir / entry.path, entry.size_bytes)
        else:
            resp = (opener or _urlopen)(entry.url, timeout)
            data = resp.read(entry.size_bytes + 1)  # bounded: one byte past pinned size
            close = getattr(resp, "close", None)
            if callable(close):
                close()
        _install(entry, data, dest)
        out[entry.path] = dest
    return out


def acquire_data(
    lock: SourceLock,
    check: RightsCheck,
    *,
    source_dir: Path | None = None,
    root: Path | None = None,
    opener: Opener | None = None,
    timeout: float = 60.0,
) -> dict[str, Path]:
    """Acquire measurement-data bytes, refused unless the rights check permits inspection."""
    return acquire(
        lock,
        source_dir=source_dir,
        root=root,
        opener=opener,
        timeout=timeout,
        roles=DATA_ROLES,
        check=check,
    )


def verify_quarantine(
    lock: SourceLock, root: Path | None = None, check: RightsCheck | None = None
) -> list[str]:
    """Re-hash quarantined evidence; data files too only with a permitting rights check."""
    roles: frozenset[FileRole] = EVIDENCE_ROLES
    if check is not None:
        require_local_inspection(check, lock)
        roles = frozenset(FileRole)
    target = (root or quarantine_root()) / lock.source_id / lock.pinned_ref
    problems = []
    for entry in (f for f in lock.files if f.role in roles):
        path = target / entry.path
        if not path.is_file() or path.is_symlink():
            problems.append(f"{entry.path}: missing from quarantine")
            continue
        problems += [f"{entry.path}: {p}" for p in check_bytes(entry, path.read_bytes())]
    return problems


def to_manifest(
    lock: SourceLock, record: RightsRecord, check: RightsCheck, *, title: str, citation: str
) -> DatasetManifest:
    """Manifest-v1 for canonical admission. Refused unless the rights check permits it."""
    require_canonical_admission(check, lock)
    return DatasetManifest(
        dataset_id=lock.source_id,
        version=f"git-{lock.pinned_ref}",
        title=title,
        data_category=DataCategory.EMPIRICAL,
        usage=Usage.CROSS_VALIDATION_ONLY,
        license=record.license_id,
        redistribution_allowed=check.redistribution_allowed,
        source=Source(
            kind="url",
            location=f"https://github.com/{lock.repository}/tree/{lock.pinned_ref}",
            citation=citation,
        ),
        files=[
            FileEntry(path=f.path, sha256=f.sha256, size_bytes=f.size_bytes, url=f.url)
            for f in lock.files
        ],
        privacy_notes=(
            f"participant consent: {record.participant_consent}; minors possible: "
            f"{record.minors_possible}; identifiers: {record.personal_identifiers_in_source}"
        ),
        handling_notes=(
            "Pinned by data/external source lock; parse only with the source-specific "
            "adapter; transform scripts are provenance and are never executed."
        ),
        audit_ref=f"docs/data-audit.md#{lock.source_id}",
    )


def admit_canonical(
    manifest: DatasetManifest,
    files_root: Path,
    *,
    store_root: Path | None = None,
    staging: Path | None = None,
) -> Path:
    """Existing store path, unchanged: ingest, re-verify, then stage read-only."""
    store.ingest_local(manifest, files_root, store_root)
    problems = store.verify(manifest, store_root)
    if problems:
        raise IntegrityError("; ".join(f"{p.path}: {p.reason}" for p in problems))
    return store.stage(manifest, store_root, staging)
