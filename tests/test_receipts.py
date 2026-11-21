"""RES-692: receipt and empirical-record verification (substance, not file existence)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from dynamisbench_soccer.qualification import empirical, receipts

SUBS = {"force_platform.single"}


def _tree(tmp_path: Path, **overrides: str) -> Path:
    files = {
        "src/dynamisbench_soccer/engines/impl.py": "def engine(x):\n    return 2 * x\n",
        "src/ref.py": "def closed_form(x):\n    return x + x\n",
        "tests/test_cmp.py": (
            "from src.ref import closed_form\n\n"
            "def test_matches():\n    assert 2 * 3 == closed_form(3)\n"
        ),
    }
    files.update(overrides)
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def _receipt(root: Path, **changes: object) -> receipts.Receipt:
    base = receipts.Receipt(
        id="R-1",
        dimension="F",
        subfamilies=("force_platform.single",),
        claim="The engine doubling equals an independently written closed form within tolerance.",
        issue="RES-675",
        implementation=("src/dynamisbench_soccer/engines/impl.py::engine",),
        reference=("src/ref.py::closed_form",),
        reference_kind="closed_form",
        independence=(
            "The closed form is written as addition in a file that never imports the engine."
        ),
        tests=("tests/test_cmp.py::test_matches",),
        tolerance=1e-9,
        tolerance_unit="dimensionless",
        tolerance_basis="exact integer arithmetic",
    )
    base = dataclasses.replace(base, **changes)
    digests = {
        ref: receipts.resolve_digest(root, ref)
        for ref in (*base.implementation, *base.reference, *base.tests)
    }
    return dataclasses.replace(base, digests={k: v for k, v in digests.items() if v})


def _errors(root: Path, r: receipts.Receipt) -> list[str]:
    return receipts.verify(r, root, SUBS)


def test_a_substantive_receipt_is_valid(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert _errors(root, _receipt(root)) == []


def test_digest_binds_the_receipt_to_the_exact_code(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    r = _receipt(root)
    (root / "src/dynamisbench_soccer/engines/impl.py").write_text(
        "def engine(x):\n    return 3 * x\n"
    )
    assert any("stale" in e for e in _errors(root, r))
    # comments and formatting do not invalidate it
    (root / "src/dynamisbench_soccer/engines/impl.py").write_text(
        "def engine(x):\n    # note\n    return 2*x\n"
    )
    assert _errors(root, r) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"claim": "short"}, "claim is not substantive"),
        ({"independence": "same"}, "independence statement"),
        ({"tolerance": 0.0}, "positive tolerance"),
        ({"tolerance_unit": " "}, "positive tolerance"),
        ({"tolerance_basis": ""}, "tolerance basis"),
        ({"check_kind": "vibes"}, "unknown check kind"),
        ({"reference_kind": "vendor_number"}, "unknown reference kind"),
        ({"dimension": "N"}, "not an M3 dimension"),
        ({"subfamilies": ("nope",)}, "unknown or missing subfamilies"),
        ({"issue": "x"}, "invalid owning issue"),
        ({"tests": ()}, "all required"),
        ({"implementation": ("tests/test_cmp.py::test_matches",)}, "must be engine, assessment"),
        (
            {"implementation": ("src/dynamisbench_soccer/engines/impl.py::missing",)},
            "does not resolve",
        ),
    ],
)
def test_receipts_with_missing_substance_are_rejected(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    root = _tree(tmp_path)
    assert any(message in e for e in _errors(root, _receipt(root, **change))), change


def test_reference_may_not_be_the_implementation_or_call_engine_code(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert any(
        "implementation file" in e
        for e in _errors(
            root, _receipt(root, reference=("src/dynamisbench_soccer/engines/impl.py::engine",))
        )
    )
    root = _tree(
        tmp_path / "b",
        **{
            "tests/test_cmp.py": (
                "from dynamisbench_soccer.engines.radar_laser import process\n\n"
                "def helper(x):\n    return process(x)\n\n"
                "def test_matches():\n    assert helper(1) == 1\n"
            )
        },
    )
    errs = _errors(root, _receipt(root, reference=("tests/test_cmp.py::helper",)))
    assert any("calls engine/assessment code" in e for e in errs)
    root = _tree(
        tmp_path / "c",
        **{
            "src/ref.py": (
                "from dynamisbench_soccer.engines import imu\n\ndef closed_form(x):\n    return x\n"
            )
        },
    )
    assert any("imports engine/assessment code" in e for e in _errors(root, _receipt(root)))


def test_skipped_xfail_and_assertion_free_tests_are_rejected(tmp_path: Path) -> None:
    for body, message in (
        ("import pytest\n\n@pytest.mark.skip\ndef test_matches():\n    assert True\n", "skipped"),
        ("import pytest\n\n@pytest.mark.xfail\ndef test_matches():\n    assert True\n", "skipped"),
        ("def test_matches():\n    x = 1\n", "no assertion"),
    ):
        root = _tree(tmp_path / message.replace(" ", "_"), **{"tests/test_cmp.py": body})
        assert any(message in e for e in _errors(root, _receipt(root))), message


def test_a_non_test_symbol_cannot_be_cited_as_the_test(tmp_path: Path) -> None:
    root = _tree(tmp_path, **{"tests/test_cmp.py": "def helper():\n    assert True\n"})
    assert any(
        "not a test function" in e
        for e in _errors(root, _receipt(root, tests=("tests/test_cmp.py::helper",)))
    )


def test_duplicate_ids_and_paths_outside_the_repo_are_rejected(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    r = _receipt(root)
    _per, global_errors = receipts.verify_all((r, r), root, SUBS)
    assert any("duplicate receipt id" in e for e in global_errors)
    escaped = _receipt(root, implementation=("src/../../etc/passwd::x",))
    assert any("does not resolve" in e or "must be engine" in e for e in _errors(root, escaped))


def test_loader_roundtrip_and_missing_file(tmp_path: Path) -> None:
    assert receipts.load(tmp_path / "none.toml") == ()
    path = tmp_path / "receipts.toml"
    path.write_text(
        '[[receipt]]\nid="R"\ndimension="F"\nsubfamilies=["a"]\nclaim="c"\nissue="RES-1"\n'
        'implementation=["src/x.py::f"]\nreference=["src/y.py::g"]\nreference_kind="closed_form"\n'
        'independence="i"\ntests=["tests/t.py::test_a"]\ntolerance=0.5\ntolerance_unit="m"\n'
        'tolerance_basis="b"\n[receipt.digests]\n"src/x.py::f"="abc"\n'
    )
    (r,) = receipts.load(path)
    assert r.tolerance == 0.5 and r.digests == {"src/x.py::f": "abc"}


def _record(**changes: object) -> empirical.EmpiricalRecord:
    base = empirical.EmpiricalRecord(
        id="E-1",
        subfamily="force_platform.single",
        dimension="B",
        instrument="Force plate model X serial 123, 1 kHz, vertical channel",
        estimand="vertical ground reaction force",
        protocol="static dead-weight loading, 5 loads, 3 repeats",
        conditions="20 C, plate bolted to slab, no cross-talk loads",
        criterion_kind="reference_standard",
        criterion="calibrated mass set traceable to national standard",
        uncertainty="expanded uncertainty U=0.4 N (k=2) from the GUM budget in the report",
        data_category="measured",
        data_sha256="a" * 64,
        rights_ref="docs/external-data/rights-clearance.md",
        n_independent_units=1,
        issue="RES-694",
    )
    return dataclasses.replace(base, **changes)


def test_measured_empirical_record_is_accepted() -> None:
    assert empirical.verify(_record(), SUBS) == []


@pytest.mark.parametrize("category", ["synthetic", "analytical", "simulated", "vendor_reported"])
def test_non_measured_data_cannot_establish_physical_validity(category: str) -> None:
    errors = empirical.verify(_record(data_category=category), SUBS)
    assert any("cannot establish physical-device validity" in e for e in errors)


@pytest.mark.parametrize(
    "change",
    [
        {"instrument": "x"},
        {"estimand": ""},
        {"protocol": "short"},
        {"conditions": ""},
        {"criterion": "tbd"},
        {"uncertainty": "small"},
        {"criterion_kind": "vendor"},
        {"data_sha256": "abc"},
        {"rights_ref": " "},
        {"n_independent_units": 0},
        {"subfamily": "imu.bogus"},
        {"issue": "x"},
        {"data_category": "mystery"},
    ],
)
def test_incomplete_empirical_records_are_rejected(change: dict[str, object]) -> None:
    assert empirical.verify(_record(**change), SUBS), change


def test_empirical_duplicate_ids_and_loader(tmp_path: Path) -> None:
    res = empirical.verify_all((_record(), _record()), SUBS)
    assert any("duplicate" in e for e in res["E-1"])
    assert empirical.load(tmp_path / "none.toml") == ()


def test_categorical_receipts_need_expected_outcomes_not_a_tolerance(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    ok = _receipt(
        root,
        check_kind="categorical",
        tolerance=0.0,
        tolerance_unit="",
        tolerance_basis="Every injected fault class is refused with its named reason.",
    )
    assert _errors(root, ok) == []
    vague = dataclasses.replace(ok, tolerance_basis="ok")
    assert any("not substantive" in e for e in _errors(root, vague))
    numeric_without_tolerance = dataclasses.replace(ok, check_kind="numeric")
    assert any("positive tolerance" in e for e in _errors(root, numeric_without_tolerance))


# ---- independent-review findings (RES-692) -------------------------------------------------

IMPL = "src/dynamisbench_soccer/engines/impl.py"
IMPL_BODY = (
    "SCALE = 2\n\ndef _helper(x):\n    return SCALE * x\n\ndef engine(x):\n    return _helper(x)\n"
)


def test_indirect_engine_use_by_a_reference_is_rejected(tmp_path: Path) -> None:
    for name, body in {
        "package import": (
            "from dynamisbench_soccer import engines\n\n"
            "def closed_form(x):\n    return engines.eng.process(x)\n"
        ),
        "agreement import": (
            "from dynamisbench_soccer.agreement.studies import run_all\n\n"
            "def closed_form(x):\n    return run_all()\n"
        ),
    }.items():
        root = _tree(tmp_path / name.replace(" ", "_"), **{"src/ref.py": body})
        assert any("engine/assessment code" in e for e in _errors(root, _receipt(root))), name
    # a test-file helper that reaches engine code through another local helper
    root = _tree(
        tmp_path / "helper",
        **{
            "tests/test_cmp.py": (
                "from dynamisbench_soccer.engines.radar_laser import process\n\n"
                "def _indirect(x):\n    return process(x)\n\n"
                "def reference(x):\n    return _indirect(x)\n\n"
                "def test_matches():\n    assert reference(1) == 1\n"
            )
        },
    )
    errs = _errors(root, _receipt(root, reference=("tests/test_cmp.py::reference",)))
    assert any("calls engine/assessment code" in e for e in errs)


def test_implementation_must_be_engine_code_and_may_not_import_its_reference(
    tmp_path: Path,
) -> None:
    root = _tree(
        tmp_path,
        **{IMPL: "from ref import closed_form\n\ndef engine(x):\n    return closed_form(x)\n"},
    )
    assert any("imports its own reference" in e for e in _errors(root, _receipt(root)))
    root = _tree(tmp_path / "b", **{"src/ref2.py": "def reference_only():\n    return 1\n"})
    bad = _receipt(root, implementation=("src/ref2.py::reference_only",))
    assert any("must be engine, assessment" in e for e in _errors(root, bad))


@pytest.mark.parametrize("body", ["assert True", "assert 1 + 1 == 2", "assert (1, 2)"])
def test_vacuous_assertions_do_not_substantiate_a_receipt(tmp_path: Path, body: str) -> None:
    root = _tree(tmp_path, **{"tests/test_cmp.py": f"def test_matches():\n    {body}\n"})
    assert any("no assertion" in e for e in _errors(root, _receipt(root)))


def test_the_digest_covers_the_same_file_helpers_and_constants_a_symbol_uses(
    tmp_path: Path,
) -> None:
    root = _tree(
        tmp_path,
        **{IMPL: IMPL_BODY},
    )
    r = _receipt(root)
    assert _errors(root, r) == []
    (root / IMPL).write_text(IMPL_BODY.replace("SCALE = 2", "SCALE = 3"))
    assert any("stale" in e for e in _errors(root, r))
    (root / IMPL).write_text(IMPL_BODY.replace("SCALE * x", "SCALE * x * 1"))
    assert any("stale" in e for e in _errors(root, r))


def test_a_duplicated_id_invalidates_every_copy(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    good = _receipt(root)
    broken = dataclasses.replace(good, dimension="K", digests={}, tests=("tests/nope.py::test_x",))
    problems, global_errors = receipts.verify_all((broken, good), root, SUBS)
    assert any("duplicate" in e for e in global_errors)
    assert problems[good.id]  # the valid copy is invalid too: nothing can hide behind an id


def test_empirical_l_needs_paired_units_and_clearance_must_resolve(tmp_path: Path) -> None:
    ok = _record(dimension="L", criterion_kind="paired_instrument", n_independent_units=20)
    assert empirical.verify(ok, SUBS) == []
    assert any(
        "at least 20" in e
        for e in empirical.verify(
            _record(dimension="L", n_independent_units=3, criterion_kind="paired_instrument"), SUBS
        )
    )
    assert any(
        "paired_instrument" in e
        for e in empirical.verify(_record(dimension="L", n_independent_units=25), SUBS)
    )
    assert any("not an M3 dimension" in e for e in empirical.verify(_record(dimension="Z"), SUBS))
    assert any(
        "must be a file under" in e
        for e in empirical.verify(_record(rights_ref="nope.md"), SUBS, tmp_path)
    )
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs/clearance.md").write_text("cleared")
    assert empirical.verify(_record(rights_ref="docs/clearance.md"), SUBS, tmp_path) == []


def test_bare_package_import_cannot_smuggle_engine_code_into_a_reference(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        **{
            "src/ref.py": (
                "import dynamisbench_soccer\n\n"
                "def closed_form(x):\n    return dynamisbench_soccer.engines.eng.process(x)\n"
            )
        },
    )
    assert any("calls engine/assessment code" in e for e in _errors(root, _receipt(root)))


def test_rights_clearance_must_be_a_repository_file_under_docs_or_data(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/ok.md").write_text("cleared")
    (tmp_path / "other.md").write_text("not a clearance")
    for bad in ("/etc/hostname", "../../etc/hostname", "other.md", "docs/missing.md"):
        errs = empirical.verify(_record(rights_ref=bad), SUBS, tmp_path)
        assert any("must be a file under" in e for e in errs), bad
    assert empirical.verify(_record(rights_ref="docs/ok.md"), SUBS, tmp_path) == []
