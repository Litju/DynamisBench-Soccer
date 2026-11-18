"""Committed records for a Zenodo-hosted external source (RES-721).

A Zenodo record is not a Git commit, so it gets its own lock kind:

* :class:`ZenodoLock` (``source-lock.json``) pins the record id and DOI, the archive
  bytes (MD5 as published plus a locally computed SHA-256 and size), and **every
  archive member** (SHA-256 and size). Members that are not data (macOS resource forks,
  editor lock files) are listed as excluded, with a reason. The lock also pins a digest
  of the record's rights-relevant metadata, so a later relicense or replaced file fails
  closed.
* :class:`ZenodoRightsRecord` (``rights.json``) is a :class:`RightsRecord` whose license
  evidence is the record metadata (``metadata.license.id``) rather than a license file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamisbench_soccer.external.models import (
    LOCK_NAME,
    RIGHTS_NAME,
    FileRole,
    RightsRecord,
    source_dir,
)

_HEX32 = r"^[0-9a-f]{32}$"
_HEX64 = r"^[0-9a-f]{64}$"
ZENODO_API = "https://zenodo.org/api/records/"
#: SPDX ids for the Zenodo license ids this project accepts as positive evidence.
ZENODO_LICENSES = {"cc-by-4.0": "CC-BY-4.0"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _relative(value: str) -> str:
    p = PurePosixPath(value)
    if not value or p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError(f"path must be relative POSIX without '..': {value!r}")
    return value


class ZenodoMember(_Strict):
    name: str
    role: Literal[FileRole.DATA, FileRole.DOCUMENTATION]
    sha256: str = Field(pattern=_HEX64)
    size_bytes: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _relative(value)


class ExcludedMember(_Strict):
    name: str
    reason: str = Field(min_length=5)


class ZenodoArchive(_Strict):
    key: str = Field(min_length=1)
    url: str
    md5: str = Field(pattern=_HEX32, description="as published by Zenodo")
    sha256: str = Field(pattern=_HEX64, description="computed locally from the bytes")
    size_bytes: int = Field(gt=0)
    members: list[ZenodoMember] = Field(min_length=1)
    excluded: list[ExcludedMember] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> ZenodoArchive:
        names = [m.name for m in self.members]
        skipped = [e.name for e in self.excluded]
        if len(names) != len(set(names)) or len(skipped) != len(set(skipped)):
            raise ValueError("duplicate archive member names")
        if set(names) & set(skipped):
            raise ValueError("a member cannot be both pinned and excluded")
        return self

    def member(self, name: str) -> ZenodoMember:
        for m in self.members:
            if m.name == name:
                return m
        raise KeyError(name)


class ZenodoLock(_Strict):
    schema_version: Literal[1] = 1
    kind: Literal["dbs-zenodo-source-lock"] = "dbs-zenodo-source-lock"
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    record_id: int = Field(gt=0)
    doi: str
    license_id: str = Field(description="Zenodo metadata.license.id at assessment")
    metadata_digest: str = Field(pattern=_HEX64, description="see metadata_subset()")
    archives: list[ZenodoArchive] = Field(min_length=1)

    @model_validator(mode="after")
    def _pinned(self) -> ZenodoLock:
        if self.doi != f"10.5281/zenodo.{self.record_id}":
            raise ValueError("doi must be the record's own Zenodo DOI")
        if self.license_id not in ZENODO_LICENSES:
            raise ValueError(f"license {self.license_id!r} is not an accepted open license")
        keys = [a.key for a in self.archives]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate archive keys")
        for a in self.archives:
            expected = f"{ZENODO_API}{self.record_id}/files/{a.key}/content"
            if a.url != expected:
                raise ValueError(f"{a.key}: url must be the record-pinned {expected}")
        return self

    @property
    def pinned_ref(self) -> str:
        return f"zenodo-{self.record_id}"

    def archive(self, key: str) -> ZenodoArchive:
        for a in self.archives:
            if a.key == key:
                return a
        raise KeyError(key)


class ZenodoRightsRecord(RightsRecord):
    """Rights record whose license evidence is the pinned Zenodo record metadata."""

    kind: Literal["dbs-zenodo-rights-record"] = "dbs-zenodo-rights-record"  # type: ignore[assignment]
    pinned_ref: str = Field(pattern=r"^zenodo-\d+$")
    license_files: list[str] = Field(default_factory=list)
    license_grant_phrase: str = Field(
        min_length=3, description="Zenodo metadata.license.id that must be on the record"
    )
    license_evidence: Literal["zenodo_record_metadata"] = "zenodo_record_metadata"

    @model_validator(mode="after")
    def _metadata_evidence(self) -> ZenodoRightsRecord:
        if self.license_files:
            raise ValueError("Zenodo license evidence is record metadata, not license files")
        spdx = ZENODO_LICENSES.get(self.license_grant_phrase)
        if spdx is None or spdx != self.license_id:
            raise ValueError("license_id must be the SPDX id of the record's license")
        return self


def metadata_subset(record: Mapping[str, Any]) -> dict[str, Any]:
    """The rights-relevant, stable part of a Zenodo record (view counts etc. excluded)."""
    m = record["metadata"]
    return {
        "id": int(record["id"]),
        "doi": record["doi"],
        "title": m["title"],
        "publication_date": m["publication_date"],
        "license": m["license"]["id"],
        "access_right": m.get("access_right"),
        "creators": [[c.get("name"), c.get("affiliation")] for c in m["creators"]],
        # free-text and access fields can carry extra terms even when license.id is unchanged
        "description": m.get("description"),
        "notes": m.get("notes"),
        "access_conditions": m.get("access_conditions"),
        "embargo_date": m.get("embargo_date"),
        "version": m.get("version"),
        "resource_type": m.get("resource_type"),
        "related_identifiers": m.get("related_identifiers"),
        "files": sorted([f["key"], int(f["size"]), f["checksum"]] for f in record["files"]),
    }


def metadata_digest(record: Mapping[str, Any]) -> str:
    text = json.dumps(metadata_subset(record), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_zenodo_lock(source_id: str, root: Path | None = None) -> ZenodoLock:
    path = source_dir(source_id, root) / LOCK_NAME
    lock = ZenodoLock.model_validate_json(path.read_text(encoding="utf-8"))
    if lock.source_id != source_id:
        raise ValueError(f"{path}: source_id {lock.source_id!r} != directory {source_id!r}")
    return lock


def load_zenodo_rights(source_id: str, root: Path | None = None) -> ZenodoRightsRecord:
    path = source_dir(source_id, root) / RIGHTS_NAME
    record = ZenodoRightsRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if record.source_id != source_id:
        raise ValueError(f"{path}: source_id {record.source_id!r} != directory {source_id!r}")
    return record


def lock_kind(source_id: str, root: Path | None = None) -> str:
    raw = json.loads((source_dir(source_id, root) / LOCK_NAME).read_text(encoding="utf-8"))
    return str(raw.get("kind", ""))
