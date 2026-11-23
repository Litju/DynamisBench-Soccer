import json
from pathlib import Path

import pytest

from dynamisbench_soccer.m4.admission import (
    DIMENSIONS,
    REQUIRED,
    Catalog,
    ReceiptBinding,
    ReviewedSlice,
    _closure,
    evaluate,
)
from dynamisbench_soccer.m4.custody import AnswerLocation
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import RegistryTask, fingerprint, load_registry

ROOT = Path(__file__).resolve().parents[1]


def _task() -> RegistryTask:
    return load_registry(ROOT / "benchmark/tasks")["elapsed-demo", 1]


def _profile(task: RegistryTask) -> ReviewedSlice:
    return ReviewedSlice(
        task_sha256=fingerprint(task),
        source=task.task.source,
        kind="analytical",
        evidence={
            role: ReceiptBinding(id="missing", sha256="0" * 64) for role in REQUIRED["analytical"]
        },
        inapplicable={role: "test-only omission" for role in DIMENSIONS - REQUIRED["analytical"]},
        implementations=("implementation",),
        oracle=("oracle",),
        files={},
        answer=AnswerLocation(id="test-only", sha256="0" * 64),
        reviewer="test-only profile",
        review_url="test-only",
        review_head="0" * 40,
    )


def _check(task: RegistryTask, profile: ReviewedSlice) -> tuple[str, tuple[str, ...]]:
    raw = Catalog(slices={task.reference_profile: profile}).model_dump_json().encode()
    decision = evaluate(ROOT, task, raw, digest(raw))
    return decision.status, decision.reasons


def test_demo_is_development_only_and_catalog_must_be_trusted() -> None:
    raw = (ROOT / "benchmark/reviewed-slices.json").read_bytes()
    task = _task()
    assert evaluate(ROOT, task, raw, digest(raw)).status == "DEVELOPMENT_ONLY_NOT_QUALIFIED"
    assert evaluate(ROOT, task, raw, "0" * 64).status == "REFUSED"


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    [
        ("methods", "id", "unknown-method", "conditions"),
        ("methods", "clock", "unrelated", "conditions"),
        ("methods", "geometry", "unsupported", "conditions"),
        ("methods", "uncertainty", "zero physical error", "conditions"),
        ("methods", "estimand", "instantaneous velocity", "conditions"),
        ("artifacts", "sha256", "0" * 64, "stale input"),
        ("artifacts", "category", "empirical", "ineligible category"),
        ("artifacts", "usage", "diagnostic", "ineligible category"),
    ],
)
def test_changed_conditions_refused(section: str, key: str, value: str, reason: str) -> None:
    original = _task()
    raw = json.loads(original.model_dump_json())
    raw[section][0][key] = value
    task = RegistryTask.model_validate_json(json.dumps(raw))
    status, reasons = _check(task, _profile(original))
    assert status == "REFUSED"
    assert reason in reasons[0]


def test_qc_required_and_wrong_subfamily_rejected() -> None:
    task = _task()
    profile = _profile(task)
    profile.evidence.pop("D")
    assert _check(task, profile)[1] == ("missing or invalid required evidence dimensions",)
    profile.evidence["D"] = ReceiptBinding(id="RCP-timing_contact-gates-D", sha256="0" * 64)
    status, reasons = _check(task, profile)
    assert status == "REFUSED"
    assert reasons == ("missing or incompatible receipt for D: RCP-timing_contact-gates-D",)


def test_closure_tracks_imported_helpers() -> None:
    closure = _closure(ROOT, {"src/dynamisbench_soccer/engines/timing_gates.py::sprint"})
    assert "src/dynamisbench_soccer/engines/event_time.py" in closure
    assert "src/dynamisbench_soccer/signals/schema.py" in closure
    assert "src/dynamisbench_soccer/data/manifest.py" in closure


@pytest.mark.parametrize(
    "kind", ["analytical", "native_signal", "protocol_assessment", "cross_device"]
)
def test_task_class_requires_explicit_omissions(kind) -> None:
    task = _task()
    profile = _profile(task).model_copy(update={"kind": kind})
    for role in REQUIRED[kind]:
        profile.evidence.setdefault(role, ReceiptBinding(id="missing", sha256="0" * 64))
    profile.inapplicable.clear()
    for role in sorted(DIMENSIONS - set(profile.evidence))[1:]:
        profile.inapplicable[role] = "independently reviewed inapplicability"
    assert _check(task, profile)[1] == ("every omitted dimension needs reviewed inapplicability",)
