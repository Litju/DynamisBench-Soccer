"""Versioned, host-side formal registry and explicit candidate projection."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from dynamisbench_soccer.m4.key_store import digest

Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Nonnegative = Annotated[float, Field(ge=0)]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def fingerprint(model: Schema) -> str:
    return digest(
        json.dumps(model.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()
    )


class Artifact(Schema):
    name: Name
    path: Text
    sha256: Sha256
    category: Literal["analytical", "synthetic", "empirical", "expert_adjudicated"]
    rights: Text
    usage: Literal["reference", "diagnostic"] = "reference"

    @model_validator(mode="after")
    def relative_path(self) -> Self:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.path:
            raise ValueError("artifact path must be normalized and relative")
        if self.path == "." or "\\" in self.path or "\x00" in self.path:
            raise ValueError("invalid artifact path")
        return self


class Method(Schema):
    id: Name
    estimand: Text
    dimension: Text
    unit: Text
    specification: dict[str, JsonValue] = Field(min_length=1)
    sampling: Text
    clock: Text
    geometry: Text
    qc: Text
    eligibility: Text
    uncertainty: Text
    absolute_tolerance: Nonnegative
    limitations: tuple[Text, ...] = Field(min_length=1)
    source_claims: tuple[Text, ...] = Field(min_length=1)


class Checkpoint(Schema):
    id: Name
    prompt: Text
    artifacts: tuple[Name, ...]
    corrects: Name | None = None


class PublicTask(Schema):
    version: Literal["m4.public-task.v1"] = "m4.public-task.v1"
    id: Name
    revision: int = Field(ge=1)
    family: Literal["A", "B", "C"]
    competencies: tuple[Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"], ...] = Field(
        min_length=1
    )
    coverage_cell: Text
    domain: Text
    protocol: Text
    source: Name
    convention: Literal["fixed", "defended"]
    allowed_tools: Literal["offline_python_stdlib"] = "offline_python_stdlib"
    answer_schema: Name


class RegistryTask(Schema):
    version: Literal["m4.registry-task.v1"] = "m4.registry-task.v1"
    task: PublicTask
    artifacts: tuple[Artifact, ...] = Field(min_length=1)
    checkpoints: tuple[Checkpoint, ...] = Field(min_length=1)
    reference_profile: Name
    methods: tuple[Method, ...] = Field(min_length=1)
    engine: Name
    grader: Name

    @model_validator(mode="after")
    def references_resolve(self) -> Self:
        available = {artifact.name for artifact in self.artifacts}
        if len(available) != len(self.artifacts):
            raise ValueError("duplicate artifact name")
        if len({method.id for method in self.methods}) != len(self.methods):
            raise ValueError("duplicate method ID")
        if self.task.convention == "fixed" and len(self.methods) != 1:
            raise ValueError("fixed tasks require exactly one method")
        seen: set[str] = set()
        released: set[str] = set()
        for checkpoint in self.checkpoints:
            if checkpoint.id in seen or (
                checkpoint.corrects is not None and checkpoint.corrects not in seen
            ):
                raise ValueError("duplicate checkpoint or invalid correction target")
            if set(checkpoint.artifacts) - available or len(set(checkpoint.artifacts)) != len(
                checkpoint.artifacts
            ):
                raise ValueError("invalid checkpoint artifacts")
            seen.add(checkpoint.id)
            released.update(checkpoint.artifacts)
        if released != available:
            raise ValueError("unreleased registry artifact")
        return self

    def candidate_payload(self, checkpoint_id: str, inputs: dict[str, bytes], seed: int) -> bytes:
        released: dict[str, str] = {}
        artifacts = {item.name: item for item in self.artifacts}
        for checkpoint in self.checkpoints:
            for name in checkpoint.artifacts:
                raw = inputs[name]
                if digest(raw) != artifacts[name].sha256:
                    raise ValueError(f"stale artifact {name}")
                released[name] = raw.decode("utf-8")
            if checkpoint.id == checkpoint_id:
                return json.dumps(
                    {
                        "task": self.task.model_dump(mode="json"),
                        "checkpoint": checkpoint.model_dump(mode="json"),
                        "inputs": released,
                        "seed": seed,
                    },
                    sort_keys=True,
                ).encode()
        raise ValueError(f"unknown checkpoint {checkpoint_id}")


def load_registry(directory: Path) -> dict[tuple[str, int], RegistryTask]:
    result = {}
    for path in sorted(directory.glob("*.json")):
        task = RegistryTask.model_validate_json(path.read_bytes())
        key = (task.task.id, task.task.revision)
        if key in result:
            raise ValueError(f"duplicate task revision {key}")
        result[key] = task
    if not result:
        raise ValueError("registry has no tasks")
    return result
