"""Fail-closed acquisition and member access for Zenodo archives (RES-721).

Order of operations, enforced in code:

1. :func:`fetch_metadata` reads the record's public metadata. It contains no
   measurement data, and is only rights evidence.
2. ``rights.verify_zenodo`` checks the metadata against the pinned digest and issues a
   sealed check.
3. :func:`acquire_archive` downloads archive bytes **only** with a check that permits
   inspection. It verifies MD5 (as published), SHA-256 and exact size, and installs the
   archive read-only in quarantine.
4. :func:`audit_archive` requires that every archive entry is a pinned member or an
   explicitly excluded one, and rejects unsafe paths, links and decompression bombs.
5. :func:`read_member` returns one pinned member's bytes **in memory**, after checking
   its size and SHA-256. Nothing is ever extracted to disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from dynamisbench_soccer.data import paths
from dynamisbench_soccer.data.manifest import (
    DataCategory,
    DatasetManifest,
    FileEntry,
    Source,
    Usage,
)
from dynamisbench_soccer.data.store import IntegrityError
from dynamisbench_soccer.external.acquire import Opener
from dynamisbench_soccer.external.rights import (
    RightsCheck,
    require_canonical_admission,
    require_local_inspection,
)
from dynamisbench_soccer.external.zenodo_models import (
    ZENODO_API,
    ZENODO_LICENSES,
    ZenodoArchive,
    ZenodoLock,
    ZenodoRightsRecord,
)

MAX_METADATA_BYTES = 5 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100.0


def _urlopen(url: str, timeout: float) -> Any:
    if not url.startswith(ZENODO_API):
        raise IntegrityError(f"refusing non-Zenodo-API url {url!r}")
    return urllib.request.urlopen(url, timeout=timeout)


def _bounded(opener: Opener | None, url: str, limit: int, timeout: float) -> bytes:
    resp = (opener or _urlopen)(url, timeout)
    try:
        data: bytes = resp.read(limit + 1)
    finally:
        close = getattr(resp, "close", None)
        if callable(close):
            close()
    if len(data) > limit:
        raise IntegrityError(f"{url}: response exceeds {limit} bytes")
    return data


def fetch_metadata(
    lock: ZenodoLock, *, opener: Opener | None = None, timeout: float = 60.0
) -> dict[str, Any]:
    """Public record metadata (rights evidence only; no measurement data)."""
    raw = _bounded(opener, f"{ZENODO_API}{lock.record_id}", MAX_METADATA_BYTES, timeout)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError("record metadata is not a JSON object")
    return value


def quarantine_dir(lock: ZenodoLock, root: Path | None = None) -> Path:
    return (root or paths.cache_home() / "quarantine") / lock.source_id / lock.pinned_ref


def _check_archive(entry: ZenodoArchive, data: bytes) -> list[str]:
    problems = []
    if len(data) != entry.size_bytes:
        problems.append(f"size {len(data)} != pinned {entry.size_bytes}")
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    if md5 != entry.md5:
        problems.append(f"md5 {md5} != published {entry.md5}")
    sha = hashlib.sha256(data).hexdigest()
    if sha != entry.sha256:
        problems.append(f"sha256 {sha[:12]}... != pinned {entry.sha256[:12]}...")
    return problems


def acquire_archive(
    lock: ZenodoLock,
    key: str,
    check: RightsCheck,
    *,
    source_dir: Path | None = None,
    root: Path | None = None,
    opener: Opener | None = None,
    timeout: float = 60.0,
) -> Path:
    """Install one pinned archive read-only in quarantine. Requires inspection rights."""
    require_local_inspection(check, lock)
    entry = lock.archive(key)
    dest = quarantine_dir(lock, root) / entry.key
    if dest.is_file() and not dest.is_symlink() and not _check_archive(entry, dest.read_bytes()):
        return dest
    if source_dir is not None:
        local = source_dir / entry.key
        info = local.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise IntegrityError(f"{local}: must be a regular file")
        if info.st_size > entry.size_bytes:
            raise IntegrityError(f"{local}: larger than pinned size")
        data = local.read_bytes()
    else:
        data = _bounded(opener, entry.url, entry.size_bytes, timeout)
    problems = _check_archive(entry, data)
    if problems:
        raise IntegrityError(f"{entry.key}: " + "; ".join(problems))
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
    return dest


def audit_archive(entry: ZenodoArchive, archive: Path) -> list[str]:
    """Every entry must be pinned or excluded; no unsafe paths, links or bombs."""
    problems: list[str] = []
    pinned = {m.name: m for m in entry.members}
    excluded = {e.name for e in entry.excluded}
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir():
                continue
            seen.add(name)
            mode = (info.external_attr >> 16) & 0o170000
            if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                problems.append(f"{name!r}: unsafe path")
            if mode == stat.S_IFLNK:
                problems.append(f"{name!r}: symbolic link")
            if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                problems.append(f"{name!r}: compression ratio exceeds {MAX_COMPRESSION_RATIO:g}")
            if name not in pinned and name not in excluded:
                problems.append(f"{name!r}: not pinned and not excluded")
            elif name in pinned and info.file_size != pinned[name].size_bytes:
                problems.append(f"{name!r}: size {info.file_size} != pinned")
    for missing in sorted((set(pinned) | excluded) - seen):
        problems.append(f"{missing!r}: listed in the lock but absent from the archive")
    return problems


def read_member(lock: ZenodoLock, key: str, archive: Path, name: str, check: RightsCheck) -> bytes:
    """One pinned member's bytes, in memory, size- and SHA-256-verified."""
    require_local_inspection(check, lock)
    entry = lock.archive(key)
    member = entry.member(name)
    if member.size_bytes > MAX_MEMBER_BYTES:
        raise IntegrityError(f"{name}: exceeds {MAX_MEMBER_BYTES} B member limit")
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(name)
        if info.file_size != member.size_bytes:
            raise IntegrityError(f"{name}: declared size {info.file_size} != pinned")
        with z.open(info) as fh:
            data = fh.read(member.size_bytes + 1)
    if len(data) != member.size_bytes or hashlib.sha256(data).hexdigest() != member.sha256:
        raise IntegrityError(f"{name}: bytes differ from the pinned member")
    return data


def to_manifest(
    lock: ZenodoLock,
    record: ZenodoRightsRecord,
    check: RightsCheck,
    *,
    title: str,
    citation: str,
    privacy_notes: str,
    handling_notes: str,
) -> DatasetManifest:
    """Manifest-v1 for canonical admission (cross-validation only), behind the rights gate."""
    require_canonical_admission(check, lock)
    return DatasetManifest(
        dataset_id=lock.source_id,
        version=lock.pinned_ref,
        title=title,
        data_category=DataCategory.EMPIRICAL,
        usage=Usage.CROSS_VALIDATION_ONLY,
        license=ZENODO_LICENSES[lock.license_id],
        redistribution_allowed=check.redistribution_allowed,
        source=Source(
            kind="url",
            location=f"https://zenodo.org/records/{lock.record_id}",
            doi=lock.doi,
            citation=citation,
        ),
        files=[
            FileEntry(path=a.key, sha256=a.sha256, size_bytes=a.size_bytes, url=a.url)
            for a in lock.archives
        ],
        privacy_notes=privacy_notes,
        handling_notes=handling_notes,
        audit_ref=f"docs/data-audit.md#{lock.source_id}",
    )
