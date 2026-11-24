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
    NATIVE_TASK_RECEIPTS,
    NATIVE_TASK_SCOPES,
    REQUIRED,
    STATISTICAL_RECEIPTS,
    Catalog,
    ReceiptBinding,
    ReviewedSlice,
    _closure,
    _oracle_closure,
    _receipt_digest,
    evaluate,
)
from dynamisbench_soccer.m4.custody import AnswerEnvelope, AnswerLocation
from dynamisbench_soccer.m4.grader import MeasurementSubmission, Submission
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.measurement import (
    CELL_OPERATION,
    PROCEDURES,
    checkpoint_methods,
    is_imu_episode,
    is_measurement_task,
)
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


def _supported(task: RegistryTask) -> bool:
    return is_measurement_task(task) or (
        task.engine == "native-lps-speed.v1"
        and (task.task.id, task.task.revision) == ("native-lps-speed", 2)
    )


def _custody_family(task: RegistryTask) -> str:
    if task.engine == "native-lps-speed.v1":
        return "native"
    if is_imu_episode(task):
        return "imu"
    if task.methods[0].id == CELL_OPERATION:
        return "cell"
    return {
        "protocol_assessment": "timing",
        "native_signal": "position",
        "cross_device": "paired",
        "statistical": "longitudinal",
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
        schema = Submission if family == "native" else MeasurementSubmission
        answer = schema.model_validate_json(json.dumps(row))
        if answer.method not in {method.id for method in checkpoint_methods(task, checkpoint)}:
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
    statistical = {receipt.id: receipt for receipt in load(ROOT / STATISTICAL_RECEIPTS)}
    native_receipts = {
        row.id: row for row in load(ROOT / f"qualification/{NATIVE_TASK_RECEIPTS}.toml")
    }
    registry = load_registry(ROOT / "benchmark/tasks")
    slices, keys = {}, {}
    for task in registry.values():
        if not _supported(task) or task.task.id not in approval["approved_tasks"]:
            continue
        method = task.methods[0]
        family = _custody_family(task)
        kind = "native_signal" if family == "native" else PROCEDURES[method.id].kind
        source = task.task.source
        selected = {}
        inapplicable = {}
        scope = NATIVE_TASK_SCOPES.get((task.task.id, task.task.revision))
        for role in sorted(DIMENSIONS):
            if role not in REQUIRED[kind]:
                inapplicable[role] = {
                    "A": "Authored statistical summaries have no sensor acquisition.",
                    "B": "The measurement model uses no device geometry or calibration stage.",
                    "C": "Named sessions supply the fixed order without hardware timing.",
                    "G": "The operation recovers no protocol event.",
                    "K": "D tests identity and design refusals. No native signal enters.",
                    "L": "One source supplies the reference under this design.",
                }[role]
                continue
            suffix = {"H": "-native-assessment", "I": "-native-bounds"}.get(role, "")
            if family == "longitudinal":
                selected[role] = statistical[f"RCP-{task.task.id}-{role}"]
                continue
            if scope is not None and role in scope[1]:
                identifier = f"RCP-{task.task.id}-{role}"
                if identifier not in native_receipts:
                    parser.error(f"missing scoped native task receipt {identifier}")
                selected[role] = native_receipts[identifier]
                continue
            if family in {"timing", "native", "cell", "imu"}:
                suffix = ""
            elif family == "paired" and role in {"L", "I"}:
                suffix = "-paired-speed"
            selected[role] = receipts[f"RCP-{source.replace('.', '-')}-{role}{suffix}"]
        production = {item for r in selected.values() for item in r.implementation}
        oracle = {item for r in selected.values() for item in r.reference}
        production_files = _closure(ROOT, production)
        oracle_files = _oracle_closure(ROOT, oracle)
        additional = (
            f"benchmark/tasks/{task.task.id}{'-v2' if family in {'native', 'imu'} else ''}.json",
            "registry/claims/device_metrology.toml",
            "registry/claims/stats_context_rtp.toml",
            "registry/sources.toml",
            "scripts/run_m5_qualification.py",
            "scripts/seal_m5_references.py",
            "scripts/m5_controls/naive.py",
        )
        if family == "imu":
            additional += (
                "benchmark/tasks/imu-frame-observability.json",
                "docs/verification/m5-res704-imu.md",
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
            kind=kind,
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

    if not slices or {
        task.task.id for task in registry.values() if task.reference_profile in slices
    } != set(approval["approved_tasks"]):
        parser.error("approval references must match the selected M5 task revisions")
    catalog = Catalog(slices=slices).model_dump_json().encode()
    pin = digest(catalog)
    for task in registry.values():
        if _supported(task) and task.reference_profile in slices:
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
