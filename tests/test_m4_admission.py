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


def test_mixed_test_oracle_tracks_only_reached_imports(tmp_path: Path) -> None:
    from dynamisbench_soccer.m4.admission import _oracle_closure

    test = tmp_path / "tests/test_mixed.py"
    test.parent.mkdir()
    test.write_text(
        "from dynamisbench_soccer.engine import process\n"
        "from dynamisbench_soccer.truth import expected\n"
        "def oracle(): return helper()\n"
        "def helper(): return expected()\n"
        "def test_agreement(): assert process() == oracle()\n"
    )
    source = tmp_path / "src/dynamisbench_soccer"
    source.mkdir(parents=True)
    for name in ("engine", "truth"):
        (source / f"{name}.py").write_text("")
    assert set(_oracle_closure(tmp_path, {"tests/test_mixed.py::oracle"})) == {
        "tests/test_mixed.py",
        "src/dynamisbench_soccer/truth.py",
    }
    test.write_text(test.read_text().replace("return expected()", "return process()"))
    assert "src/dynamisbench_soccer/engine.py" in _oracle_closure(
        tmp_path, {"tests/test_mixed.py::oracle"}
    )


def test_statistical_task_cannot_borrow_native_review_or_depart_from_analytical_inputs():
    task = load_registry(ROOT / "benchmark/tasks")["lps-repeat-session-change", 1]
    review = _profile(task)
    mismatch = ("incompatible statistical task or source category",)
    assert _check(task, review.model_copy(update={"kind": "native_signal"})) == (
        "REFUSED",
        mismatch,
    )
    wrong = _task()
    wrong_review = _profile(wrong).model_copy(update={"kind": "statistical"})
    assert _check(wrong, wrong_review) == ("REFUSED", mismatch)
    wrong_artifact = task.artifacts[0].model_copy(update={"category": "synthetic"})
    wrong = task.model_copy(update={"artifacts": (wrong_artifact, *task.artifacts[1:])})
    wrong_review = _profile(wrong).model_copy(update={"kind": "statistical"})
    assert _check(wrong, wrong_review) == ("REFUSED", mismatch)
