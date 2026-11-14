"""Reproducible cleanup policy (ADR-0001 rule 12, docs/data-policy.md).

Targets
-------
* ``artifacts``  - benchmark run outputs: ``<cache>/runs`` and in-repo ``var/``, ``runs/``
* ``workspaces`` - staging trees and agent/test workspaces: ``<cache>/staging``,
  ``<cache>/workspaces``
* ``caches``     - in-repo build/tool caches: ``__pycache__``, ``.pytest_cache``,
  ``.mypy_cache``, ``.ruff_cache``, ``dist``, ``build``

The content-addressed **store is never touched** by ``clean``. Removing data from
it is an explicit, separate action (``dbs data gc``, future).

Only paths inside the known roots are deleted, and symlinks are unlinked, never
followed.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from dynamisbench_soccer.data import paths

TARGETS = ("artifacts", "workspaces", "caches")
_CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_CACHE_TOP_LEVEL = {"dist", "build"}
_SKIP_DIRS = {".git", ".venv", "node_modules"}


@dataclass(frozen=True)
class Candidate:
    target: str
    path: Path
    size_bytes: int


def _size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(p.lstat().st_size for p in path.rglob("*") if not p.is_dir())


def _children(root: Path) -> Iterator[Path]:
    if root.is_dir():
        yield from sorted(root.iterdir())


def _walk_caches(repo: Path) -> Iterator[Path]:
    for name in sorted(_CACHE_TOP_LEVEL):
        if (repo / name).is_dir():
            yield repo / name
    stack = [repo]
    while stack:
        current = stack.pop()
        for child in sorted(current.iterdir()):
            if not child.is_dir() or child.is_symlink() or child.name in _SKIP_DIRS:
                continue
            if child.name in _CACHE_DIR_NAMES:
                yield child
            else:
                stack.append(child)


def plan(
    targets: Iterable[str],
    repo: Path,
    older_than_hours: float | None = None,
) -> list[Candidate]:
    """List what ``clean`` would remove, without touching anything."""
    selected = set(targets)
    unknown = selected - set(TARGETS)
    if unknown:
        raise ValueError(f"unknown clean targets: {sorted(unknown)}")
    found: list[tuple[str, Path]] = []
    if "artifacts" in selected:
        for root in (paths.runs_root(), repo / "var", repo / "runs"):
            found += [("artifacts", p) for p in _children(root)]
    if "workspaces" in selected:
        for root in (paths.staging_root(), paths.workspaces_root()):
            found += [("workspaces", p) for p in _children(root)]
    if "caches" in selected:
        found += [("caches", p) for p in _walk_caches(repo)]

    cutoff = None if older_than_hours is None else time.time() - older_than_hours * 3600
    result = []
    for target, path in found:
        if cutoff is not None and path.lstat().st_mtime > cutoff:
            continue
        result.append(Candidate(target, path, _size(path)))
    return result


def apply(candidates: Iterable[Candidate]) -> int:
    """Delete planned candidates. Returns bytes freed."""
    freed = 0
    for cand in candidates:
        freed += cand.size_bytes
        if cand.path.is_symlink() or cand.path.is_file():
            cand.path.unlink(missing_ok=True)
        elif cand.path.is_dir():
            shutil.rmtree(cand.path)
    return freed
