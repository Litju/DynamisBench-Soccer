"""Canonical-environment guard (ADR-0001).

Fails when the repository, the virtual environment or the interpreter resolve to a
Windows-mounted filesystem (``/mnt/<drive>``) or to a Windows interpreter. Mixing
Windows and Linux Python, venvs or caches is forbidden by ADR-0001 rule 11.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_WINDOWS_MOUNT = re.compile(r"^/mnt/[a-zA-Z](/|$)")

#: Soft memory budget for a full benchmark run (ADR-0001 rule 8).
MEMORY_BUDGET_BYTES = 8 * 1024**3


@dataclass
class EnvReport:
    """Result of an environment check."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def is_windows_mount(path: Path | str) -> bool:
    """Return True when ``path`` lives on a WSL Windows drive mount such as ``/mnt/c``."""
    return bool(_WINDOWS_MOUNT.match(str(Path(path).resolve())))


def _is_wsl() -> bool:
    return "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ


def check_environment(repo_root: Path | None = None) -> EnvReport:
    """Check the running environment against ADR-0001."""
    report = EnvReport()
    root = (repo_root or Path.cwd()).resolve()
    executable = Path(sys.executable)
    prefix = Path(sys.prefix)

    report.facts.update(
        {
            "platform": sys.platform,
            "python": platform.python_version(),
            "wsl": str(_is_wsl()),
            "repo_root": str(root),
            "interpreter": str(executable),
            "venv": str(prefix),
        }
    )

    if not sys.platform.startswith("linux"):
        report.errors.append(
            f"platform {sys.platform!r} is not the canonical scientific runtime (Linux/WSL2)"
        )
    if executable.suffix.lower() == ".exe":
        report.errors.append(f"Windows interpreter in use: {executable}")
    for label, path in (("repository", root), ("interpreter", executable), ("venv", prefix)):
        if is_windows_mount(path):
            report.errors.append(f"{label} is on a Windows mount ({path}); use ~/projects")
    uv_cache = os.environ.get("UV_CACHE_DIR")
    if uv_cache and is_windows_mount(uv_cache):
        report.errors.append(f"UV_CACHE_DIR on a Windows mount ({uv_cache})")

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = pages * page_size
        report.facts["memory_total_gib"] = f"{total / 1024**3:.1f}"
        if total < MEMORY_BUDGET_BYTES:
            report.warnings.append(
                "host memory below the 8 GiB budget; large episodes may need --max-memory"
            )
    except (ValueError, OSError):  # pragma: no cover - non-POSIX
        report.warnings.append("could not determine physical memory")

    return report
