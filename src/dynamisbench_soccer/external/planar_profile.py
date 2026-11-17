"""Planar (2D) external-validation profile, version 1 (RES-721).

A narrow profile for **non-soccer instrument-validation data** that cannot meet the
schema-v2 ``BundleMeta`` contract without fabrication: there is no Z coordinate, no
registered soccer ``test_id``, no session date and no clock. Schema v2 is not weakened
for it. Instead the profile states exactly what is known and fixes the unknowns as
literal values:

* ``dimensionality`` is always ``"2D"``, and ``z_coordinate`` is always
  ``"not_recorded"``. There is no field that could hold a synthesised Z.
* ``soccer_task`` is always ``False``, and there is no ``test_id`` field.
* ``timestamps`` is ``"absent"`` and ``pairing`` is ``"row_index_within_session"``: the
  only time relation is the depositor's row order. That is not clock evidence.
* ``units`` and each system's frame are ``"undeclared"`` unless the source declares
  them. Comparability is a separate, computed gate, never a profile field.
* ``usage`` is always ``"cross-validation-only"``.

The JSON Schema is committed at ``docs/schemas/planar-validation-profile.v1.json``.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisbench_soccer.external.models import SignalStage

PROFILE_VERSION: Literal["dbs-planar-validation@1"] = "dbs-planar-validation@1"
SCHEMA_DOC = "docs/schemas/planar-validation-profile.v1.json"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanarSystem(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    role: Literal["device_estimate", "depositor_reference"]
    stage: SignalStage
    x_column: str
    y_column: str
    frame: Literal["undeclared"] | str = "undeclared"
    meaning: str = Field(min_length=10)

    @model_validator(mode="after")
    def _not_native(self) -> PlanarSystem:
        if self.stage is SignalStage.PIPELINE_DERIVED:
            raise ValueError("source positions cannot be pipeline-derived")
        return self


class PlanarSession(_Strict):
    scenario: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    session_id: int = Field(ge=1)
    n_rows: int = Field(ge=1)
    first_row: int = Field(ge=0, description="0-based data-row index in the source member")


class PlanarProfile(_Strict):
    profile: Literal["dbs-planar-validation@1"] = PROFILE_VERSION
    source_id: str
    member: str
    member_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter: str
    dimensionality: Literal["2D"] = "2D"
    z_coordinate: Literal["not_recorded"] = "not_recorded"
    soccer_task: Literal[False] = False
    usage: Literal["cross-validation-only"] = "cross-validation-only"
    timestamps: Literal["absent"] = "absent"
    pairing: Literal["row_index_within_session"] = "row_index_within_session"
    sampling_rate: Literal["undeclared"] = "undeclared"
    units: Literal["undeclared"] = "undeclared"
    units_note: str = Field(min_length=10)
    systems: list[PlanarSystem] = Field(min_length=2)
    sessions: list[PlanarSession] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> PlanarProfile:
        ids = [s.id for s in self.systems]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate system ids")
        roles = [s.role for s in self.systems]
        if roles.count("depositor_reference") != 1:
            raise ValueError("exactly one depositor_reference system is required")
        keys = [(s.scenario, s.session_id) for s in self.sessions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate scenario/session")
        return self


def json_schema() -> str:
    return json.dumps(PlanarProfile.model_json_schema(), indent=2, sort_keys=True) + "\n"
