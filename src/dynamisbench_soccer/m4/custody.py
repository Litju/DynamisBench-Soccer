"""Read host-only, salted answer envelopes from an external private directory."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import Name, Schema, Sha256

MAX_KEY_BYTES = 64 * 1024


class AnswerLocation(Schema):
    version: Literal["m4.answer-location.v1"] = "m4.answer-location.v1"
    id: Name
    sha256: Sha256


class AnswerEnvelope(Schema):
    version: Literal["m4.private-answer.v1"] = "m4.private-answer.v1"
    task_sha256: Sha256
    nonce: Sha256
    checkpoints: dict[Name, dict[str, JsonValue]] = Field(min_length=1)


def load_private(
    root: Path, repository: Path, location: AnswerLocation, task_sha256: str
) -> AnswerEnvelope:
    root = root.resolve(strict=True)
    repository = repository.resolve(strict=True)
    if root.is_relative_to(repository) or repository.is_relative_to(root):
        raise ValueError("private custody must be disjoint from the checkout")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(directory)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("custody directory must be owner-private")
        filename = f"{location.id}.json"
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise ValueError("answer must be an owner-private regular file with one link")
            raw = stream.read(MAX_KEY_BYTES + 1)
    finally:
        os.close(directory)
    if len(raw) > MAX_KEY_BYTES or digest(raw) != location.sha256:
        raise ValueError("private answer exceeds size limit or has a stale digest")
    answer = AnswerEnvelope.model_validate_json(raw)
    if answer.task_sha256 != task_sha256:
        raise ValueError("private answer belongs to another task")
    return answer


def check_projection(payload: bytes, answer: AnswerEnvelope, location: AnswerLocation) -> None:
    if answer.nonce.encode() in payload or location.sha256.encode() in payload:
        raise ValueError("private answer metadata leaked into candidate payload")
