"""Content-addressed dataset store and staging.

One physical copy of every file lives in the store, keyed by SHA-256
(``<store>/ab/abcdef...``). Worktrees, venvs and staging directories *link* to it,
so data is never duplicated (ADR-0001 rule 5). Every operation verifies hashes and
fails closed on a mismatch.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dynamisbench_soccer.data import paths
from dynamisbench_soccer.data.manifest import DatasetManifest, FileEntry, sha256_file


class IntegrityError(RuntimeError):
    """Raised when bytes on disk do not match the manifest."""


@dataclass(frozen=True)
class Problem:
    path: str
    reason: str


def blob_path(sha256: str, store: Path | None = None) -> Path:
    root = store or paths.store_root()
    return root / sha256[:2] / sha256


def _check_blob(entry: FileEntry, blob: Path) -> Problem | None:
    if not blob.is_file():
        return Problem(entry.path, "missing from store")
    size = blob.stat().st_size
    if size != entry.size_bytes:
        return Problem(entry.path, f"size {size} != manifest {entry.size_bytes}")
    actual = sha256_file(blob)
    if actual != entry.sha256:
        return Problem(entry.path, f"sha256 {actual[:12]}... != manifest {entry.sha256[:12]}...")
    return None


def verify(manifest: DatasetManifest, store: Path | None = None) -> list[Problem]:
    """Re-hash every stored file of ``manifest``; return all problems found."""
    problems = []
    for entry in manifest.files:
        problem = _check_blob(entry, blob_path(entry.sha256, store))
        if problem:
            problems.append(problem)
    return problems


def _install_blob(entry: FileEntry, src: Path, store: Path | None) -> None:
    dest = blob_path(entry.sha256, store)
    if dest.is_file() and _check_blob(entry, dest) is None:
        return  # already present and intact
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".incoming-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as inp:
            shutil.copyfileobj(inp, out, length=1024 * 1024)
        problem = _check_blob(entry, tmp)
        if problem:
            raise IntegrityError(f"{entry.path}: {problem.reason}")
        tmp.chmod(0o444)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def ingest_local(manifest: DatasetManifest, source_dir: Path, store: Path | None = None) -> None:
    """Copy files from a local directory into the store, verifying each one."""
    for entry in manifest.files:
        _install_blob(entry, source_dir / entry.path, store)


def fetch(manifest: DatasetManifest, store: Path | None = None, timeout: float = 60.0) -> None:
    """Download files that declare a ``url`` into the store, verifying each one."""
    for entry in manifest.files:
        if _check_blob(entry, blob_path(entry.sha256, store)) is None:
            continue
        if not entry.url:
            raise IntegrityError(f"{entry.path}: not in store and no url to fetch from")
        with tempfile.TemporaryDirectory(prefix="dbs-fetch-") as tmp:
            download = Path(tmp) / "download"
            with (
                urllib.request.urlopen(entry.url, timeout=timeout) as resp,
                download.open("wb") as out,
            ):
                shutil.copyfileobj(resp, out, length=1024 * 1024)
            _install_blob(entry, download, store)


def stage(
    manifest: DatasetManifest,
    store: Path | None = None,
    staging: Path | None = None,
) -> Path:
    """Verify, then expose a dataset as a temporary read-only tree of symlinks.

    Returns the staging directory. The caller (or ``dbs clean --workspaces``) removes it.
    """
    problems = verify(manifest, store)
    if problems:
        detail = "; ".join(f"{p.path}: {p.reason}" for p in problems)
        raise IntegrityError(f"{manifest.key} failed verification: {detail}")
    root = staging or paths.staging_root()
    root.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(dir=root, prefix=f"{manifest.dataset_id}-{manifest.version}-"))
    for entry in manifest.files:
        link = target / entry.path
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob_path(entry.sha256, store))
    (target / "MANIFEST.json").write_text(manifest.to_json(), encoding="utf-8")
    return target
