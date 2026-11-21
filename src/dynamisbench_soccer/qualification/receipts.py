"""Independent computational-verification receipts (RES-692 / RES-675, Policy A).

A receipt is a machine-checked claim that one qualification dimension of one or more
subfamilies was verified computationally: it names the implementation symbols under test, an
independent reference (a closed form or an independent implementation), the tests that compare
them with a declared tolerance, and content digests of every cited symbol. A file existing, or
an evidence-kind string, is never enough. A receipt establishes computational-reference
correctness under declared conditions only. It can never establish physical-device validity
(that needs measured evidence, see ``empirical.py``) or scientific qualification (M8).

Checks are structural and content-bound, not a substitute for independent review of whether
the cited test really exercises the claim; ``tests/test_receipts_execute.py`` executes the
cited tests so a receipt can never rest on a failing or skipped test.
"""

from __future__ import annotations

import ast
import hashlib
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DIMENSIONS_M3 = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "O")
REFERENCE_KINDS = frozenset(
    {"closed_form", "independent_implementation", "published_worked_example"}
)
# A reference must not be computed by the code it verifies.
FORBIDDEN_REFERENCE_IMPORTS = ("dynamisbench_soccer.engines", "dynamisbench_soccer.assessment")
SKIP_MARKERS = frozenset({"skip", "skipif", "xfail"})
MIN_CLAIM_CHARS = 60


@dataclass(frozen=True)
class Receipt:
    id: str
    dimension: str
    subfamilies: tuple[str, ...]
    claim: str
    issue: str
    implementation: tuple[str, ...]
    reference: tuple[str, ...]
    reference_kind: str
    independence: str
    tests: tuple[str, ...]
    tolerance: float
    tolerance_unit: str
    tolerance_basis: str
    digests: dict[str, str] = field(default_factory=dict)
    # "numeric": agreement within a positive tolerance. "categorical": the expected verdict,
    # refusal or classification is asserted exactly; the basis states the expected outcomes.
    check_kind: str = "numeric"


def load(path: Path) -> tuple[Receipt, ...]:
    if not path.is_file():
        return ()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in raw.get("receipt", []):
        out.append(
            Receipt(
                id=row["id"],
                dimension=row["dimension"],
                subfamilies=tuple(row["subfamilies"]),
                claim=row["claim"],
                issue=row["issue"],
                implementation=tuple(row["implementation"]),
                reference=tuple(row["reference"]),
                reference_kind=row["reference_kind"],
                independence=row["independence"],
                tests=tuple(row["tests"]),
                tolerance=float(row.get("tolerance", 0.0)),
                tolerance_unit=row.get("tolerance_unit", ""),
                tolerance_basis=row["tolerance_basis"],
                digests=dict(row.get("digests", {})),
                check_kind=row.get("check_kind", "numeric"),
            )
        )
    return tuple(out)


def _split(ref: str) -> tuple[str, str]:
    path, sep, symbol = ref.partition("::")
    if not sep or not path or not symbol:
        raise ValueError(f"reference {ref!r} must be 'path::Symbol'")
    return path, symbol


def _find(tree: ast.Module, symbol: str) -> ast.AST | None:
    """Module-level function/class/assignment, or ``Class.method``."""
    scope: list[ast.stmt] = tree.body
    node: ast.AST | None = None
    for part in symbol.split("."):
        node = None
        for stmt in scope:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if stmt.name == part:
                    node = stmt
                    break
            elif isinstance(stmt, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == part for t in stmt.targets):
                    node = stmt
                    break
            elif (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == part
            ):
                node = stmt
                break
        if node is None:
            return None
        scope = node.body if isinstance(node, ast.ClassDef) else []
    return node


def digest_node(node: ast.AST) -> str:
    """Digest of the symbol's syntax tree: stable under comments and formatting only."""
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _stamp(target: Path) -> tuple[int, int]:
    st = target.stat()
    return (st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=512)
def _parse_cached(target: str, stamp: tuple[int, int]) -> ast.Module:
    return ast.parse(Path(target).read_text(encoding="utf-8"))


def _parse(root: Path, path: str) -> ast.Module | None:
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return None
    return _parse_cached(str(target), _stamp(target))


@dataclass(frozen=True)
class SymbolInfo:
    digest: str
    names: frozenset[str]
    banned_imports: frozenset[str]
    marks: frozenset[str]
    asserts: bool


@lru_cache(maxsize=4096)
def _info_cached(target: str, stamp: tuple[int, int], symbol: str) -> SymbolInfo | None:
    tree = _parse_cached(target, stamp)
    node = _find(tree, symbol)
    if node is None:
        return None
    return SymbolInfo(
        digest_node(node),
        frozenset(n.id for n in ast.walk(node) if isinstance(n, ast.Name)),
        frozenset(_engine_names(tree)),
        frozenset(_marks(node)),
        _has_assertion(node),
    )


def symbol_info(root: Path, ref: str) -> SymbolInfo | None:
    path, symbol = _split(ref)
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return None
    return _info_cached(str(target), _stamp(target), symbol)


def resolve_digest(root: Path, ref: str) -> str | None:
    info = symbol_info(root, ref)
    return None if info is None else info.digest


def _engine_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(FORBIDDEN_REFERENCE_IMPORTS):
                names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(FORBIDDEN_REFERENCE_IMPORTS):
                    names.add((a.asname or a.name).split(".")[0])
    return names


def _marks(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for dec in getattr(node, "decorator_list", []):
        for sub in ast.walk(dec):
            if isinstance(sub, ast.Attribute) and sub.attr in SKIP_MARKERS:
                found.add(sub.attr)
    return found


ASSERTING_CALLS = frozenset({"approx", "raises", "allclose", "isclose", "array_equal", "warns"})


def _has_assertion(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            return True
        name = None
        if isinstance(sub, ast.Attribute):
            name = sub.attr
        elif isinstance(sub, ast.Name):
            name = sub.id
        if name is not None and (name in ASSERTING_CALLS or name.startswith("assert_")):
            return True
    return False


def verify(receipt: Receipt, root: Path, subfamily_ids: set[str]) -> list[str]:
    """Return every reason this receipt does not establish what it claims (empty = valid)."""
    rid = receipt.id
    errors: list[str] = []
    if receipt.dimension not in DIMENSIONS_M3:
        errors.append(f"{rid}: dimension {receipt.dimension!r} is not an M3 dimension")
    if not receipt.subfamilies or set(receipt.subfamilies) - subfamily_ids:
        errors.append(f"{rid}: unknown or missing subfamilies")
    if len(receipt.claim.strip()) < MIN_CLAIM_CHARS:
        errors.append(f"{rid}: claim is not substantive")
    if len(receipt.independence.strip()) < MIN_CLAIM_CHARS:
        errors.append(f"{rid}: independence statement is not substantive")
    if not receipt.issue.startswith("RES-"):
        errors.append(f"{rid}: invalid owning issue")
    if receipt.reference_kind not in REFERENCE_KINDS:
        errors.append(f"{rid}: unknown reference kind {receipt.reference_kind!r}")
    if receipt.check_kind not in {"numeric", "categorical"}:
        errors.append(f"{rid}: unknown check kind {receipt.check_kind!r}")
    if receipt.check_kind == "numeric" and (
        not (receipt.tolerance > 0) or not receipt.tolerance_unit.strip()
    ):
        errors.append(f"{rid}: a positive tolerance with a unit is required")
    if len(receipt.tolerance_basis.strip()) < 20:
        errors.append(f"{rid}: tolerance basis (or expected outcomes) is not substantive")
    if not receipt.implementation or not receipt.reference or not receipt.tests:
        errors.append(f"{rid}: implementation, reference and tests are all required")
        return errors
    impl_files = set()
    for ref in receipt.implementation:
        try:
            path, _ = _split(ref)
        except ValueError as exc:
            errors.append(f"{rid}: {exc}")
            continue
        impl_files.add(path)
        if not path.startswith("src/"):
            errors.append(f"{rid}: implementation {ref} must live under src/")
    for ref in (*receipt.implementation, *receipt.reference, *receipt.tests):
        try:
            current = resolve_digest(root, ref)
        except ValueError:
            continue  # already reported
        if current is None:
            errors.append(f"{rid}: {ref} does not resolve to a symbol")
        elif receipt.digests.get(ref) != current:
            errors.append(f"{rid}: digest of {ref} is missing or stale")
    for ref in receipt.reference:
        try:
            path, _symbol = _split(ref)
            info = symbol_info(root, ref)
        except ValueError:
            continue
        if path in impl_files:
            errors.append(f"{rid}: reference {ref} lives in an implementation file")
        if info is None:
            continue
        used = info.names & info.banned_imports
        if used:
            errors.append(f"{rid}: reference {ref} calls engine/assessment code {sorted(used)}")
        if path.startswith("src/") and info.banned_imports:
            errors.append(f"{rid}: reference file {path} imports engine/assessment code")
    for ref in receipt.tests:
        try:
            path, symbol = _split(ref)
            info = symbol_info(root, ref)
        except ValueError:
            continue
        if info is None:
            continue
        if not path.startswith("tests/") or not symbol.split(".")[-1].startswith("test_"):
            errors.append(f"{rid}: {ref} is not a test function under tests/")
        if info.marks:
            errors.append(f"{rid}: {ref} is skipped or expected to fail")
        if not info.asserts:
            errors.append(f"{rid}: {ref} contains no assertion")
    return errors


def verify_all(
    receipts: tuple[Receipt, ...], root: Path, subfamily_ids: set[str]
) -> tuple[dict[str, list[str]], list[str]]:
    """(errors per receipt id, global errors such as duplicate ids)."""
    problems: dict[str, list[str]] = {}
    global_errors: list[str] = []
    seen: set[str] = set()
    for r in receipts:
        if r.id in seen:
            global_errors.append(f"duplicate receipt id {r.id}")
        seen.add(r.id)
        problems[r.id] = verify(r, root, subfamily_ids)
    return problems, global_errors
