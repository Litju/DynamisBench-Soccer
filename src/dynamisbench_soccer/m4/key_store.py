"""DEVELOPMENT_ONLY_NOT_QUALIFIED grader-side verification of private answer bytes."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import Field

from dynamisbench_soccer.m4.contracts import Answer, Contract, Episode


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


class PrivateKey(Contract):
    episode_sha256: str
    answer: Answer
    alternative_methods: tuple[str, ...] = ()
    atol: float = Field(ge=0)


def load_key(path: Path, expected_sha256: str, episode: Episode) -> PrivateKey:
    raw = path.read_bytes()
    if digest(raw) != expected_sha256:
        raise ValueError("private answer digest mismatch")
    key = PrivateKey.model_validate_json(raw)
    if key.episode_sha256 != digest(episode.model_dump_json().encode()):
        raise ValueError("private answer belongs to another episode")
    return key
