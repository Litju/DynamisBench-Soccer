"""Bind independently approved M5 references to the M4 operator catalog."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
from pathlib import Path

from dynamisbench_soccer.m4.admission import (
    DIMENSIONS,
    REQUIRED,
    Catalog,
    ReceiptBinding,
    ReviewedSlice,
    _closure,
    _oracle_closure,
    _receipt_digest,
    evaluate,
)
from dynamisbench_soccer.m4.custody import AnswerEnvelope, AnswerLocation
from dynamisbench_soccer.m4.grader import MeasurementSubmission
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.measurement import ENGINE, PROCEDURES
from dynamisbench_soccer.m4.task_registry import RegistryTask, fingerprint, load_registry
from dynamisbench_soccer.qualification.receipts import load

ROOT = Path(__file__).resolve().parents[1]


def _read_private(directory: Path, name: str) -> bytes:
    path = directory / name
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 0o077
    ):
        raise ValueError("review inputs must be owner-private regular files")
    return path.read_bytes()


def _write_private(path: Path, raw: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)


def _custody_family(task: RegistryTask) -> str:
    return {
        "protocol_assessment": "timing",
        "native_signal": "position",
        "cross_device": "paired",
    }[PROCEDURES[task.methods[0].id].kind]


def _expected(task, family, directory, approval):
    raw = _read_private(directory, f"{family}-expected.json")
    oracle = _read_private(directory, f"{family}-oracle.py")
    provenance = _read_private(directory, f"{family}-provenance.md")
    if approval["input_sha256"][family] != {
        "expected": digest(raw),
        "oracle": digest(oracle),
        "provenance": digest(provenance),
    }:
        raise ValueError("private reference inputs differ from reviewed bytes")
    expected = json.loads(raw)[task.task.id]
    if set(expected) != {checkpoint.id for checkpoint in task.checkpoints}:
        raise ValueError("independent answer does not cover every checkpoint")
    checkpoints = {}
    for checkpoint, row in expected.items():
        answer = MeasurementSubmission.model_validate_json(json.dumps(row))
        if answer.method != task.methods[0].id:
            raise ValueError("independent answer specifies the wrong method")
        checkpoints[checkpoint] = {
            answer.method: {
                "decision": answer.decision,
                "reasons": [answer.reason],
                "selected_trials": list(answer.selected_trials),
                "required_citations": list(answer.citations),
                "allowed_citations": list(answer.citations),
                "required_claims": list(answer.claims),
                "allowed_claims": list(answer.claims),
                "quantity": answer.quantity.model_dump() if answer.quantity else None,
                "operands": [item.model_dump() for item in answer.operands],
                "corrects": answer.corrects,
            }
        }
    return AnswerEnvelope(
        task_sha256=fingerprint(task), nonce=secrets.token_hex(32), checkpoints=checkpoints
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("custody", type=Path)
    args = parser.parse_args()
    private = args.custody.resolve(strict=True)
    if (
        private.is_relative_to(ROOT)
        or ROOT.is_relative_to(private)
        or private.stat().st_uid != os.getuid()
        or private.stat().st_mode & 0o077
    ):
        parser.error("custody must be a disjoint owner-private directory")

    approval = json.loads(_read_private(private, "review-attestation.json"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (
        approval["head"] != head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip()
    ):
        parser.error("attestation must cover the exact clean head")
    receipts = {receipt.id: receipt for receipt in load(ROOT / "qualification/receipts.toml")}
    registry = load_registry(ROOT / "benchmark/tasks")
    slices, keys = {}, {}
    for task in registry.values():
        if task.engine != ENGINE or task.task.id not in approval["approved_tasks"]:
            continue
        method = task.methods[0]
        procedure = PROCEDURES[method.id]
        family = _custody_family(task)
        selected = {}
        inapplicable = {}
        for role in sorted(DIMENSIONS):
            if role not in REQUIRED[procedure.kind]:
                inapplicable[role] = {
                    "G": "The fixed sample or paired fixed samples recover no protocol event.",
                    "L": "This single-source slice refuses equivalence without a paired study.",
                }[role]
                continue
            suffix = {"H": "-native-assessment", "I": "-native-bounds"}.get(role, "")
            if family == "timing":
                suffix = ""
            elif family == "paired" and role in {"L", "I"}:
                suffix = "-paired-speed"
            selected[role] = receipts[f"RCP-{procedure.source.replace('.', '-')}-{role}{suffix}"]
        production = {item for r in selected.values() for item in r.implementation}
        oracle = {item for r in selected.values() for item in r.reference}
        production_files = _closure(ROOT, production)
        oracle_files = _oracle_closure(ROOT, oracle)
        additional = (
            f"benchmark/tasks/{task.task.id}.json",
            "registry/claims/device_metrology.toml",
            "registry/claims/stats_context_rtp.toml",
            "registry/sources.toml",
            "scripts/run_m5_qualification.py",
            "scripts/seal_m5_references.py",
            "scripts/m5_controls/naive.py",
        )
        bound = {
            path.relative_to(ROOT).as_posix()
            for directory in ("src", "tests")
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        } | {"uv.lock", "pyproject.toml", *additional}
        envelope = _expected(task, family, private, approval)
        key_raw = envelope.model_dump_json().encode()
        answer = AnswerLocation(id=secrets.token_hex(16), sha256=digest(key_raw))
        keys[answer.id] = key_raw
        slices[task.reference_profile] = ReviewedSlice(
            task_sha256=fingerprint(task),
            source=task.task.source,
            kind=procedure.kind,
            evidence={
                role: ReceiptBinding(id=r.id, sha256=_receipt_digest(r))
                for role, r in selected.items()
            },
            inapplicable=inapplicable,
            implementations=tuple(sorted(production)),
            oracle=tuple(sorted(oracle)),
            shared_infrastructure=tuple(sorted(production_files.keys() & oracle_files.keys())),
            additional_files=additional,
            files=production_files
            | oracle_files
            | {path: digest((ROOT / path).read_bytes()) for path in sorted(bound)},
            answer=answer,
            private_inputs_sha256=approval["input_sha256"][family],
            reviewer=approval["reviewer"],
            review_url=approval["review_url"],
            review_head=head,
        )

    if not slices or set(slices) != set(approval["approved_tasks"]):
        parser.error("approval references must match the selected M5 task revisions")
    catalog = Catalog(slices=slices).model_dump_json().encode()
    pin = digest(catalog)
    for task in registry.values():
        if task.reference_profile in slices:
            result = evaluate(ROOT, task, catalog, pin)
            if result.status != "ENGINEERING_ADMITTED":
                parser.error(f"{task.task.id}: {'; '.join(result.reasons)}")
    for name, raw in keys.items():
        _write_private(private / f"{name}.json", raw)
    catalog_path = private / f"catalog-{pin}.json"
    _write_private(catalog_path, catalog)
    print(pin)


if __name__ == "__main__":
    main()
