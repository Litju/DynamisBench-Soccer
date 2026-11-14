"""Locations outside the repository for data, staging and run artifacts.

All roots are overridable by environment variables so CI and tests can isolate them:

* ``DBS_DATA_HOME``  - shared content-addressed store
  (default ``$XDG_DATA_HOME/dynamisbench-soccer``)
* ``DBS_CACHE_HOME`` - staging directories and run workspaces
  (default ``$XDG_CACHE_HOME/dynamisbench-soccer``)
"""

from __future__ import annotations

import os
from pathlib import Path

APP = "dynamisbench-soccer"


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else Path.home() / fallback


def data_home() -> Path:
    explicit = os.environ.get("DBS_DATA_HOME")
    return Path(explicit) if explicit else _xdg("XDG_DATA_HOME", ".local/share") / APP


def cache_home() -> Path:
    explicit = os.environ.get("DBS_CACHE_HOME")
    return Path(explicit) if explicit else _xdg("XDG_CACHE_HOME", ".cache") / APP


def store_root() -> Path:
    """Single shared store: one copy of each file, addressed by SHA-256."""
    return data_home() / "store" / "sha256"


def staging_root() -> Path:
    return cache_home() / "staging"


def workspaces_root() -> Path:
    return cache_home() / "workspaces"


def runs_root() -> Path:
    return cache_home() / "runs"
