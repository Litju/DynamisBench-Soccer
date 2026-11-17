"""Committed, reviewable records for one external source (RES-720).

Two records live in Git per source, under ``data/external/<source_id>/``:

* :class:`SourceLock` (``source-lock.json``) - the exact upstream bytes: a pinned Git
  commit, and per file the SHA-256, size and Git blob SHA-1. URLs must embed the
  pinned commit, so a moving branch can never be fetched.
* :class:`RightsRecord` (``rights.json``) - the rights and privacy determination.
  It keeps *software/package license* evidence apart from rights to the *original
  measurement data* and participant consent, and it fails validation if a record
  claims canonical admission or redistribution without the evidence for it.

Neither record is a :class:`~dynamisbench_soccer.data.manifest.DatasetManifest`. A
manifest is created only when the rights record permits canonical admission.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EXTERNAL_ROOT = Path("data/external")
LOCK_NAME = "source-lock.json"
RIGHTS_NAME = "rights.json"

_SOURCE_ID = r"^[a-z0-9][a-z0-9-]{2,63}$"
_HEX40 = r"^[0-9a-f]{40}$"
_HEX64 = r"^[0-9a-f]{64}$"
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SignalStage(StrEnum):
    """Processing stage of a signal. Stages are never merged or relabelled."""

    MEASURED_NATIVE = "measured_native"  # sensor samples as acquired by the device
    MANUFACTURER_PROCESSED = "manufacturer_processed"  # vendor software output
    PACKAGE_DERIVED = "package_derived"  # transform by a third-party package
    PIPELINE_DERIVED = "pipeline_derived"  # computed here, by dbs, for diagnostics


class FileRole(StrEnum):
    DATA = "data"
    LICENSE = "license"
    DOCUMENTATION = "documentation"
    TRANSFORM_SCRIPT = "transform_script"  # recorded for provenance, never executed


class PinnedFile(_Strict):
    path: str
    role: FileRole
    sha256: str = Field(pattern=_HEX64)
    size_bytes: int = Field(ge=0)
    git_blob_sha1: str = Field(pattern=_HEX40, description="Git object id; not a SHA-256")
    url: str

    @field_validator("path")
    @classmethod
    def _relative(cls, value: str) -> str:
        p = PurePosixPath(value)
        if not value or p.is_absolute() or ".." in p.parts or "\\" in value:
            raise ValueError(f"path must be relative POSIX without '..': {value!r}")
        return value


class SourceLock(_Strict):
    schema_version: Literal[1] = 1
    kind: Literal["dbs-external-source-lock"] = "dbs-external-source-lock"
    source_id: str = Field(pattern=_SOURCE_ID)
    repository: str = Field(description="GitHub owner/name")
    pinned_ref: str = Field(pattern=_HEX40, description="full commit SHA, never a branch")
    files: list[PinnedFile] = Field(min_length=1)

    @model_validator(mode="after")
    def _pinned(self) -> SourceLock:
        if not _REPO.match(self.repository):
            raise ValueError("repository must be owner/name")
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate file paths in source lock")
        for f in self.files:
            expected = (
                f"https://raw.githubusercontent.com/{self.repository}/{self.pinned_ref}/{f.path}"
            )
            if f.url != expected:
                raise ValueError(f"{f.path}: url must be the commit-pinned {expected}")
        roles = {f.role for f in self.files}
        if FileRole.DATA not in roles or FileRole.LICENSE not in roles:
            raise ValueError("a source lock needs at least one data and one license file")
        return self

    def file(self, path: str) -> PinnedFile:
        for f in self.files:
            if f.path == path:
                return f
        raise KeyError(path)

    def by_role(self, role: FileRole) -> list[PinnedFile]:
        return [f for f in self.files if f.role is role]


class RightsStatus(StrEnum):
    VERIFIED = "verified"
    PENDING = "pending"
    REFUSED = "refused"


class RightsRecord(_Strict):
    """Source-specific rights determination. Records evidence; never infers approval."""

    schema_version: Literal[1] = 1
    kind: Literal["dbs-external-rights-record"] = "dbs-external-rights-record"
    source_id: str = Field(pattern=_SOURCE_ID)
    pinned_ref: str = Field(pattern=_HEX40)
    assessed_on: str
    assessed_by: str = Field(min_length=1)

    license_id: str = Field(min_length=1, description="SPDX id of the license file found")
    license_files: list[str] = Field(min_length=1)
    license_grant_phrase: str = Field(min_length=10, description="must occur in a license file")
    copyright_holder: str = Field(min_length=1, description="must occur in a license file")
    license_scope: str = Field(min_length=10)
    dataset_specific_terms: str = Field(min_length=1)

    measurement_data_rights: RightsStatus
    participant_consent: Literal["documented", "not_documented", "not_applicable"]
    minors_possible: bool
    personal_identifiers_in_source: Literal["none", "present", "unknown"]

    local_inspection_permitted: bool
    local_inspection_basis: str = Field(min_length=10)
    canonical_admission: RightsStatus
    redistribution_allowed: bool
    permitted_outputs: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    evidence_urls: list[str] = Field(min_length=1)

    @field_validator("assessed_on")
    @classmethod
    def _date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _consistent(self) -> RightsRecord:
        if self.canonical_admission is RightsStatus.VERIFIED:
            if self.measurement_data_rights is not RightsStatus.VERIFIED:
                raise ValueError("canonical admission requires verified measurement-data rights")
            if self.participant_consent == "not_documented":
                raise ValueError("canonical admission requires documented participant consent")
            if self.personal_identifiers_in_source != "none":
                raise ValueError("canonical admission requires no personal identifiers")
            if self.blockers:
                raise ValueError("a verified admission cannot list open blockers")
        elif not self.blockers:
            raise ValueError("pending or refused admission must state its blockers")
        if self.redistribution_allowed and (
            self.canonical_admission is not RightsStatus.VERIFIED or self.minors_possible
        ):
            raise ValueError(
                "redistribution needs verified admission and no possible minors (privacy rule 5)"
            )
        if self.local_inspection_permitted and self.canonical_admission is RightsStatus.REFUSED:
            raise ValueError("a refused source may not be inspected locally")
        return self


def source_dir(source_id: str, root: Path | None = None) -> Path:
    return (root or EXTERNAL_ROOT) / source_id


def load_lock(source_id: str, root: Path | None = None) -> SourceLock:
    path = source_dir(source_id, root) / LOCK_NAME
    lock = SourceLock.model_validate_json(path.read_text(encoding="utf-8"))
    if lock.source_id != source_id:
        raise ValueError(f"{path}: source_id {lock.source_id!r} != directory {source_id!r}")
    return lock


def load_rights(source_id: str, root: Path | None = None) -> RightsRecord:
    path = source_dir(source_id, root) / RIGHTS_NAME
    record = RightsRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if record.source_id != source_id:
        raise ValueError(f"{path}: source_id {record.source_id!r} != directory {source_id!r}")
    return record


def source_ids(root: Path | None = None) -> list[str]:
    base = root or EXTERNAL_ROOT
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
