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
FORBIDDEN_REFERENCE_IMPORTS = (
    "dynamisbench_soccer.engines",
    "dynamisbench_soccer.assessment",
    "dynamisbench_soccer.agreement",
)
FORBIDDEN_PACKAGES = frozenset({"engines", "assessment", "agreement"})
IMPLEMENTATION_PACKAGES = (
    "src/dynamisbench_soccer/engines/",
    "src/dynamisbench_soccer/assessment/",
    "src/dynamisbench_soccer/agreement/",
)
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


def _module_level(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out[stmt.name] = stmt
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = stmt
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out[stmt.target.id] = stmt
    return out


def closure(tree: ast.Module, node: ast.AST) -> dict[str, ast.AST]:
    """Module-level definitions in the same file that the symbol (transitively) uses."""
    defs = _module_level(tree)
    seen: dict[str, ast.AST] = {}
    todo = [node]
    while todo:
        current = todo.pop()
        for sub in ast.walk(current):
            if isinstance(sub, ast.Name) and sub.id in defs and sub.id not in seen:
                if defs[sub.id] is node:
                    continue
                seen[sub.id] = defs[sub.id]
                todo.append(defs[sub.id])
    return seen


def digest_node(node: ast.AST, tree: ast.Module | None = None) -> str:
    """Digest of the symbol and of every same-file definition it transitively uses.

    Stable under comments and formatting only. Code reached through other modules is not
    covered here; ``tests/test_receipts_execute.py`` runs the cited tests to catch that.
    """
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode())
    if tree is not None:
        for name, dep in sorted(closure(tree, node).items()):
            digest.update(name.encode())
            digest.update(ast.dump(dep, include_attributes=False).encode())
    return digest.hexdigest()


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
    reached = [node, *closure(tree, node).values()]
    return SymbolInfo(
        digest_node(node, tree),
        frozenset(n.id for part in reached for n in ast.walk(part) if isinstance(n, ast.Name)),
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
            elif node.module == "dynamisbench_soccer":
                names.update(a.asname or a.name for a in node.names if a.name in FORBIDDEN_PACKAGES)
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


def _substantive(expr: ast.AST) -> bool:
    """An assertion about nothing computed (``assert True``, ``assert 1 + 1 == 2``) is vacuous."""
    return any(
        isinstance(n, ast.Name | ast.Call | ast.Attribute | ast.Subscript) for n in ast.walk(expr)
    )


def _has_assertion(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert) and _substantive(sub.test):
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
        if not path.startswith(IMPLEMENTATION_PACKAGES):
            errors.append(
                f"{rid}: implementation {ref} must be engine, assessment or agreement code"
            )
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
    reference_modules = set()
    for ref in receipt.reference:
        try:
            ref_path = _split(ref)[0]
        except ValueError:
            continue
        if ref_path.startswith("src/") and ref_path.endswith(".py"):
            reference_modules.add(ref_path[4:-3].replace("/", "."))
    for path in sorted(impl_files):
        if path.startswith("src/dynamisbench_soccer/agreement/"):
            # Agreement studies are harnesses: they build inputs from the substrate and compare
            # engine outputs, so importing it is their purpose, not a circular reference.
            continue
        tree = _parse(root, path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in reference_modules:
                        errors.append(f"{rid}: implementation {path} imports its own reference")
            elif module in reference_modules:
                errors.append(f"{rid}: implementation {path} imports its own reference")
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
    """(errors per receipt id, global errors such as duplicate ids).

    Every receipt that shares an id is invalid: a broken copy can never hide behind, or be
    counted as, a valid one.
    """
    problems: dict[str, list[str]] = {}
    global_errors: list[str] = []
    counts: dict[str, int] = {}
    for r in receipts:
        counts[r.id] = counts.get(r.id, 0) + 1
    for r in receipts:
        errs = verify(r, root, subfamily_ids)
        if counts[r.id] > 1:
            errs = [*errs, f"{r.id}: duplicate receipt id"]
        problems.setdefault(r.id, []).extend(errs)
    global_errors.extend(f"duplicate receipt id {rid}" for rid, n in counts.items() if n > 1)
    return problems, global_errors
