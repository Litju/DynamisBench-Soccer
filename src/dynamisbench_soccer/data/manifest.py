"""Versioned external dataset manifest (schema version 1).

A manifest records *where* a dataset comes from, *what* it is allowed to be used for
and the exact bytes of every file (SHA-256 + size). Manifests live in Git under
``data/manifests/``. The data itself never does.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANIFEST_SCHEMA_VERSION: Literal[1] = 1
_HASH_CHUNK = 1024 * 1024


class DataCategory(StrEnum):
    """Data categories are kept separate everywhere (project governance)."""

    EMPIRICAL = "empirical"
    ANALYTICAL = "analytical"
    SYNTHETIC = "synthetic"
    EXPERT_ADJUDICATED = "expert-adjudicated"


class Usage(StrEnum):
    """What the benchmark may do with a dataset (decided by the data audit, RES-666)."""

    TASK_INPUT = "task-input"
    CROSS_VALIDATION_ONLY = "cross-validation-only"
    EXCLUDED = "excluded"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["url", "local", "generated"]
    location: str = Field(min_length=1, description="URL, absolute path, or generator id")
    doi: str | None = None
    citation: str | None = None


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(description="POSIX relative path inside the dataset")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    url: str | None = Field(default=None, description="direct download URL, if any")

    @field_validator("path")
    @classmethod
    def _relative_posix(cls, value: str) -> str:
        p = PurePosixPath(value)
        if p.is_absolute() or ".." in p.parts or "\\" in value or not value:
            raise ValueError(f"file path must be relative POSIX without '..': {value!r}")
        return value


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: str = Field(min_length=1)
    title: str
    data_category: DataCategory
    usage: Usage
    license: str = Field(description="SPDX id or explicit license statement")
    redistribution_allowed: bool
    source: Source
    files: list[FileEntry] = Field(default_factory=list)
    privacy_notes: str = ""
    handling_notes: str = Field(default="", description="safe-loading / conversion requirements")
    audit_ref: str | None = Field(default=None, description="e.g. docs/data-audit.md#anchor")

    @model_validator(mode="after")
    def _consistency(self) -> DatasetManifest:
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate file paths in manifest")
        if self.usage is not Usage.EXCLUDED and not self.files:
            raise ValueError("non-excluded datasets must list their files")
        return self

    @property
    def key(self) -> str:
        return f"{self.dataset_id}@{self.version}"

    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    # --- serialization ------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 in 1 MiB chunks (bounded memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def describe_directory(root: Path) -> list[FileEntry]:
    """Hash every regular file below ``root`` into sorted manifest entries."""
    entries = []
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = file.relative_to(root).as_posix()
        entries.append(
            FileEntry(path=rel, sha256=sha256_file(file), size_bytes=file.stat().st_size)
        )
    return entries
