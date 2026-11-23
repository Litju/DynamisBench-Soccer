import json
import secrets
from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamisbench_soccer.m4.custody import (
    AnswerEnvelope,
    AnswerLocation,
    check_projection,
    load_private,
)
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import RegistryTask, fingerprint, load_registry

ROOT = Path(__file__).resolve().parents[1]


def test_registry_and_checkpoint_projection(tmp_path: Path) -> None:
    task = load_registry(ROOT / "benchmark/tasks")["elapsed-demo", 1]
    inputs = {item.name: (ROOT / item.path).read_bytes() for item in task.artifacts}
    first = task.candidate_payload("initial", inputs, 1)
    assert b"elapsed-correction" not in first
    assert b"reference_profile" not in first
    assert (
        set(json.loads(task.candidate_payload("correction", inputs, 1))["inputs"]) == inputs.keys()
    )
    assert fingerprint(task) == fingerprint(
        RegistryTask.model_validate_json(task.model_dump_json())
    )
    with pytest.raises(ValueError, match="stale artifact"):
        task.candidate_payload("initial", {"elapsed-demo": b"changed"}, 1)
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(task.model_dump_json())
    with pytest.raises(ValueError, match="duplicate task"):
        load_registry(tmp_path)
    raw = json.loads(task.model_dump_json())
    raw["checkpoints"][0]["corrects"] = "correction"
    with pytest.raises(ValidationError, match="correction target"):
        RegistryTask.model_validate_json(json.dumps(raw))


def test_external_ephemeral_key_is_host_only(tmp_path: Path) -> None:
    task = load_registry(ROOT / "benchmark/tasks")["elapsed-demo", 1]
    task_hash = fingerprint(task)
    nonce = secrets.token_hex(32)
    envelope = AnswerEnvelope(task_sha256=task_hash, nonce=nonce, checkpoints={"initial": {"v": 1}})
    raw = envelope.model_dump_json().encode()
    location = AnswerLocation(id="key", sha256=digest(raw))
    path = tmp_path / "key.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    assert load_private(tmp_path, ROOT, location, task_hash) == envelope
    payload = task.candidate_payload(
        "initial", {"elapsed-demo": (ROOT / task.artifacts[0].path).read_bytes()}, 42
    )
    check_projection(payload, envelope, location)
    with pytest.raises(ValueError, match="leaked"):
        check_projection(payload + nonce.encode(), envelope, location)
    linked = tmp_path / "linked.json"
    linked.hardlink_to(path)
    with pytest.raises(ValueError, match="one link"):
        load_private(tmp_path, ROOT, location, task_hash)
    linked.unlink()
    linked.symlink_to(path)
    with pytest.raises(OSError):
        load_private(tmp_path, ROOT, AnswerLocation(id="linked", sha256=digest(raw)), task_hash)


def test_malformed_custody_error_omits_private_values(tmp_path: Path) -> None:
    raw = b'{"checkpoints":{"initial":{"value":"PRIVATE_VALUE"}}}'
    path = tmp_path / "key.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    with pytest.raises(ValueError, match=r"^invalid private answer envelope$") as error:
        load_private(tmp_path, ROOT, AnswerLocation(id="key", sha256=digest(raw)), "0" * 64)
    assert error.value.__suppress_context__
