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
Kind = Literal["analytical", "native_signal", "protocol_assessment", "cross_device", "statistical"]
COMPUTATION = frozenset({"D", "E", "F", "H", "I", "J", "M", "O"})
NATIVE = COMPUTATION | {"A", "B", "C", "K"}
REQUIRED: dict[Kind, frozenset[str]] = {
    "analytical": COMPUTATION,
    "statistical": COMPUTATION,
    "native_signal": NATIVE,
    "protocol_assessment": NATIVE | {"G"},
    "cross_device": NATIVE | {"L"},
}
DIMENSIONS = frozenset(receipts.DIMENSIONS_M3)
PACKAGE = "dynamisbench_soccer"
SOURCE = f"src/{PACKAGE}/"
CELL_TASK = ("load-cell-net-torque-ramp", 1)
CELL_TASK_SHA256 = "133670a4fa6026efe72bb52a434fbf64e06ab2e5e82f98c2fd53bb953cb17949"
IMU_TASK = ("imu-frame-observability", 2)
IMU_TASK_SHA256 = "e78372488e6195411bbef0b89091d1999ad22fa30c6215d7bd54bd2ad8096d06"
NATIVE_TASK_RECEIPTS = "native_task_receipts"
NATIVE_TASK_SCOPES = {
    CELL_TASK: (CELL_TASK_SHA256, frozenset({"C", "E", "H", "I", "K"})),
    IMU_TASK: (IMU_TASK_SHA256, frozenset({"A", "C", "G", "H", "M"})),
}
STATISTICAL_TASK = ("lps-repeat-session-change", 1)
STATISTICAL_RECEIPTS = "qualification/task_receipts.toml"


class ReceiptBinding(Schema):
    id: Name
    sha256: Sha256


class ReviewedSlice(Schema):
    task_sha256: Sha256
    source: Name
    kind: Kind
    evidence: dict[str, ReceiptBinding]
    inapplicable: dict[str, Text]
    implementations: tuple[Text, ...] = Field(min_length=1)
    oracle: tuple[Text, ...] = Field(min_length=1)
    shared_infrastructure: tuple[Text, ...] = ()
    additional_files: tuple[Text, ...] = ()
    files: dict[str, Sha256]
    answer: AnswerLocation
    private_inputs_sha256: dict[Name, Sha256] = Field(default_factory=dict)
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


def _import_modules(node: ast.Import | ast.ImportFrom, package: str) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    base = resolve_name("." * node.level + (node.module or ""), package)
    return {base, *(f"{base}.{alias.name}" for alias in node.names)}


def _package_paths(root: Path, modules: set[str]) -> set[str]:
    paths: set[str] = set()
    for module in modules:
        if module != PACKAGE and not module.startswith(f"{PACKAGE}."):
            continue
        parts = module.split(".")
        for length in range(1, len(parts) + 1):
            prefix = "src/" + "/".join(parts[:length])
            paths.update(
                candidate
                for candidate in (f"{prefix}.py", f"{prefix}/__init__.py")
                if (root / candidate).is_file()
            )
    return paths


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
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules.update(_import_modules(node, package))
        pending.update(_package_paths(root, modules | {package}) - found.keys())
    return found


def _oracle_closure(root: Path, entries: set[str]) -> dict[str, str]:
    """Follow the oracle symbols inside mixed test modules, including their used imports."""
    source_entries = {entry for entry in entries if entry.startswith(SOURCE)}
    test_files: dict[str, str] = {}
    for entry in entries - source_entries:
        path, symbol = receipts._split(entry)
        if (
            not path.startswith("tests/test_")
            or not path.endswith(".py")
            or Path(path).parent != Path("tests")
        ):
            raise ValueError(f"oracle outside supported reference paths: {path}")
        raw = _read(root, path)
        tree = ast.parse(raw, filename=path)
        target = receipts._find(tree, symbol)
        if target is None:
            raise ValueError(f"missing oracle symbol {symbol}")
        nodes = [
            part
            for definition in (target, *receipts.closure(tree, target).values())
            for part in ast.walk(definition)
        ]
        names = {node.id for node in nodes if isinstance(node, ast.Name)}
        imports = [node for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom))]
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) and any(
                alias.name == "*" or (alias.asname or alias.name.split(".")[0]) in names
                for alias in stmt.names
            ):
                imports.append(stmt)
        for stmt in imports:
            modules = _import_modules(stmt, "")
            if any(
                module.startswith(("test_", "tests.")) or module == "tests" for module in modules
            ):
                raise ValueError("test oracle imports another test module")
            source_entries.update(_package_paths(root, modules))
        test_files[path] = digest(raw)
    return _closure(root, source_entries) | test_files


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
        statistical = review.kind == "statistical"
        if statistical != ((task.task.id, task.task.revision) == STATISTICAL_TASK) or (
            statistical and any(artifact.category != "analytical" for artifact in task.artifacts)
        ):
            return decision("REFUSED", "incompatible statistical task or source category")
        identity = (task.task.id, task.task.revision)
        scope = NATIVE_TASK_SCOPES.get(identity)
        if task.task.id == IMU_TASK[0] and identity != IMU_TASK:
            return decision("REFUSED", "IMU development revision is outside the scored scope")
        if scope is not None and (task_hash != scope[0] or review.kind != "protocol_assessment"):
            device = "load-cell" if identity == CELL_TASK else "IMU"
            return decision("REFUSED", f"incompatible frozen native {device} task or review kind")
        roles = set(review.evidence)
        if not REQUIRED[review.kind] <= roles <= DIMENSIONS:
            return decision("REFUSED", "missing or invalid required evidence dimensions")
        if set(review.inapplicable) != DIMENSIONS - roles:
            return decision("REFUSED", "every omitted dimension needs reviewed inapplicability")

        rows = receipts.load(
            root / (STATISTICAL_RECEIPTS if statistical else "qualification/receipts.toml")
        )
        native_rows = (
            receipts.load(root / f"qualification/{NATIVE_TASK_RECEIPTS}.toml") if scope else ()
        )
        counts = Counter(row.id for row in (*rows, *native_rows))
        inventory = {row.id: row for row in rows}
        native_inventory = {row.id: row for row in native_rows}
        audit_document = json.loads(_read(root, "qualification/receipt_audit.json"))
        if not isinstance(audit_document, dict) or not isinstance(
            audit := audit_document.get("task_receipts" if statistical else "receipts"), dict
        ):
            return decision("REFUSED", "invalid Policy A audit map")
        implementations: set[str] = set()
        oracles: set[str] = set()
        for role, binding in sorted(review.evidence.items()):
            identifier = binding.id
            scoped = scope is not None and role in scope[1]
            row = (native_inventory if scoped else inventory).get(identifier)
            if (
                row is None
                or counts[identifier] != 1
                or row.dimension != role
                or review.source not in row.subfamilies
                or ((statistical or scoped) and identifier != f"RCP-{task.task.id}-{role}")
                or (scoped and row.issue != "RES-704")
            ):
                return decision(
                    "REFUSED", f"missing or incompatible receipt for {role}: {identifier}"
                )
            current = _receipt_digest(row)
            evidence[identifier] = current
            if current != binding.sha256:
                return decision("REFUSED", f"receipt changed since task review: {identifier}")
            table = audit_document.get(NATIVE_TASK_RECEIPTS) if scoped else audit
            record = table.get(identifier, {}) if isinstance(table, dict) else {}
            if (
                not isinstance(record, dict)
                or record.get("verdict") != "SUPPORTED"
                or record.get("bound_digest") != current
            ):
                return decision("REFUSED", f"review absent or stale: {identifier}")
            errors = receipts.verify(
                row, root, set(row.subfamilies), statistical_task=statistical, native_task=scoped
            )
            if errors:
                return decision("REFUSED", *errors)
            implementations.update(row.implementation)
            oracles.update(row.reference)

        if set(review.implementations) != implementations or set(review.oracle) != oracles:
            return decision("REFUSED", "incomplete or incompatible implementation/oracle closure")
        production = _closure(root, implementations)
        oracle = _oracle_closure(root, set(review.oracle))
        production_entries = {ref.split("::", 1)[0] for ref in implementations}
        oracle_entries = {ref.split("::", 1)[0] for ref in oracles}
        shared = production.keys() & oracle.keys()
        if (
            production_entries & oracle.keys()
            or oracle_entries & production.keys()
            or shared != set(review.shared_infrastructure)
        ):
            return decision("REFUSED", "reference and implementation are circular")
        # Tests can select fixtures dynamically and import helpers anywhere under src/.
        bound_files = {
            path.relative_to(root).as_posix()
            for directory in ("tests", "src")
            for path in (root / directory).rglob("*")
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
