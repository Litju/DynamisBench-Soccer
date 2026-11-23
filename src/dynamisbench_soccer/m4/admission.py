"""Fail-closed, task-scoped admission against an operator-pinned review catalog."""

from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import asdict
from importlib.util import resolve_name
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from dynamisbench_soccer.m4.custody import AnswerLocation
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import (
    Name,
    RegistryTask,
    Schema,
    Sha256,
    Text,
    fingerprint,
)
from dynamisbench_soccer.qualification import receipts

Status = Literal["DEVELOPMENT_ONLY_NOT_QUALIFIED", "ENGINEERING_ADMITTED", "REFUSED"]
# These roles describe evidence needed by this reference, not a subfamily gate.
REQUIRED = {"D", "E", "F", "H", "I", "J", "M", "O"}
RAW = {"A", "B", "C", "G", "K"}
PACKAGE = "dynamisbench_soccer"
SOURCE = f"src/{PACKAGE}/"


class ReceiptBinding(Schema):
    id: Name
    sha256: Sha256


class ReviewedSlice(Schema):
    task_sha256: Sha256
    source: Name
    kind: Literal["specified_events", "native_signal"]
    evidence: dict[str, ReceiptBinding]
    implementations: tuple[Text, ...] = Field(min_length=1)
    oracle: tuple[Text, ...] = Field(min_length=1)
    shared_infrastructure: tuple[Text, ...] = ()
    additional_files: tuple[Text, ...] = ()
    files: dict[str, Sha256]
    answer: AnswerLocation
    reviewer: Text
    review_url: Text
    review_head: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class Catalog(Schema):
    version: Literal["m4.reviewed-slices.v1"] = "m4.reviewed-slices.v1"
    slices: dict[Name, ReviewedSlice]


class Admission(Schema):
    version: Literal["m4.admission.v1"] = "m4.admission.v1"
    status: Status
    task_sha256: Sha256
    catalog_sha256: Sha256
    reasons: tuple[str, ...] = ()
    reviewed_slice: ReviewedSlice | None = None
    evidence: dict[str, Sha256] = Field(default_factory=dict)


def _read(root: Path, relative: str) -> bytes:
    path = root / relative
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved != path or not resolved.is_file():
        raise ValueError(f"invalid evidence path {relative}")
    return resolved.read_bytes()


def _closure(root: Path, entries: set[str]) -> dict[str, str]:
    """Overapproximate static imports, including package initialization and unused branches."""
    pending = {entry.split("::", 1)[0] for entry in entries}
    found: dict[str, str] = {}
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.startswith(SOURCE) or not path.endswith(".py"):
            raise ValueError(f"dependency outside supported package: {path}")
        raw = _read(root, path)
        found[path] = digest(raw)
        module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        package = module.rpartition(".")[0]
        modules: set[str] = set()
        for node in ast.walk(ast.parse(raw, filename=path)):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = resolve_name("." * node.level + (node.module or ""), package)
                modules.add(base)
                modules.update(f"{base}.{alias.name}" for alias in node.names)
        modules.add(package)
        for module in modules:
            if module != PACKAGE and not module.startswith(f"{PACKAGE}."):
                continue
            parts = module.split(".")
            for length in range(1, len(parts) + 1):
                prefix = "src/" + "/".join(parts[:length])
                for candidate in (f"{prefix}.py", f"{prefix}/__init__.py"):
                    if (root / candidate).is_file() and candidate not in found:
                        pending.add(candidate)
    return found


def _receipt_digest(receipt: receipts.Receipt) -> str:
    # Matches scripts/bind_audit.py and the canonical Policy A audit artifact.
    return digest(json.dumps(asdict(receipt), sort_keys=True).encode())


def evaluate(
    root: Path, task: RegistryTask, catalog_bytes: bytes, trusted_catalog_sha256: str
) -> Admission:
    """The trust anchor comes from deployment configuration, never the task or candidate."""
    task_hash = fingerprint(task)
    catalog_hash = digest(catalog_bytes)
    review = None
    evidence: dict[str, str] = {}

    def decision(status: Status, *reasons: str) -> Admission:
        return Admission(
            status=status,
            task_sha256=task_hash,
            catalog_sha256=catalog_hash,
            reviewed_slice=review,
            evidence=evidence,
            reasons=reasons,
        )

    try:
        root = root.resolve(strict=True)
        if catalog_hash != trusted_catalog_sha256:
            return decision("REFUSED", "untrusted review catalog")
        for artifact in task.artifacts:
            if (
                artifact.category not in {"analytical", "synthetic"}
                or artifact.usage != "reference"
            ):
                return decision("REFUSED", f"ineligible category or usage: {artifact.name}")
            if digest(_read(root, artifact.path)) != artifact.sha256:
                return decision("REFUSED", f"stale input: {artifact.name}")
        review = Catalog.model_validate_json(catalog_bytes).slices.get(task.reference_profile)
        if review is None:
            return decision("DEVELOPMENT_ONLY_NOT_QUALIFIED", "reference slice lacks review")
        if review.task_sha256 != task_hash or review.source != task.task.source:
            return decision("REFUSED", "task conditions or source differ from reviewed slice")
        required = REQUIRED | (RAW if review.kind == "native_signal" else set())
        if not required <= review.evidence.keys():
            return decision("REFUSED", "missing required evidence dimensions")

        rows = receipts.load(root / "qualification/receipts.toml")
        counts = Counter(row.id for row in rows)
        inventory = {row.id: row for row in rows}
        audit_document = json.loads(_read(root, "qualification/receipt_audit.json"))
        if not isinstance(audit_document, dict) or not isinstance(
            audit := audit_document.get("receipts"), dict
        ):
            return decision("REFUSED", "invalid Policy A audit map")
        implementations: set[str] = set()
        oracles: set[str] = set()
        for role, binding in sorted(review.evidence.items()):
            identifier = binding.id
            row = inventory.get(identifier)
            if (
                row is None
                or counts[identifier] != 1
                or row.dimension != role
                or review.source not in row.subfamilies
            ):
                return decision(
                    "REFUSED", f"missing or incompatible receipt for {role}: {identifier}"
                )
            current = _receipt_digest(row)
            evidence[identifier] = current
            if current != binding.sha256:
                return decision("REFUSED", f"receipt changed since task review: {identifier}")
            record = audit.get(identifier, {})
            if (
                not isinstance(record, dict)
                or record.get("verdict") != "SUPPORTED"
                or record.get("bound_digest") != current
            ):
                return decision("REFUSED", f"review absent or stale: {identifier}")
            errors = receipts.verify(row, root, set(row.subfamilies))
            if errors:
                return decision("REFUSED", *errors)
            implementations.update(row.implementation)
            oracles.update(row.reference)

        if set(review.implementations) != implementations or set(review.oracle) != oracles:
            return decision("REFUSED", "incomplete or incompatible implementation/oracle closure")
        production = _closure(root, implementations)
        oracle = _closure(root, set(review.oracle))
        production_entries = {ref.split("::", 1)[0] for ref in implementations}
        oracle_entries = {ref.split("::", 1)[0] for ref in oracles}
        shared = production.keys() & oracle.keys()
        if (
            production_entries & oracle.keys()
            or oracle_entries & production.keys()
            or shared != set(review.shared_infrastructure)
        ):
            return decision("REFUSED", "reference and implementation are circular")
        # Test helpers may be selected dynamically by pytest, so bind the whole test tree.
        bound_files = {
            path.relative_to(root).as_posix()
            for path in (root / "tests").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        } | {"uv.lock", "pyproject.toml", *review.additional_files}
        bound_files.update(ref.split("::", 1)[0] for ref in oracles if not ref.startswith(SOURCE))
        actual_files = (
            production | oracle | {path: digest(_read(root, path)) for path in bound_files}
        )
        if actual_files != review.files:
            return decision("REFUSED", "stale or incomplete transitive file closure")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SyntaxError,
        ImportError,
        ValidationError,
    ) as exc:
        return decision("REFUSED", f"invalid evidence: {exc}")
    return decision("ENGINEERING_ADMITTED")
