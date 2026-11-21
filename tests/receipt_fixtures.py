"""Receipt fixtures over real repository symbols, for gate-logic tests only (RES-692).

These are not verification evidence for any subfamily: they let tests build in-memory ledgers
whose every M3 dimension carries a *structurally valid* receipt, so the gate's decision rules
can be exercised with valid, invalid, adversarial and incomplete inputs.
"""

from __future__ import annotations

from pathlib import Path

from dynamisbench_soccer.qualification import receipts as receipts_mod
from dynamisbench_soccer.qualification.ledger import M3_DIMENSIONS, Ledger
from dynamisbench_soccer.qualification.receipts import Receipt

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = "src/dynamisbench_soccer/engines/radar_laser.py::process"
REFERENCE = "src/dynamisbench_soccer/reference/motion_models.py::MonoExponentialSprint"
TEST = (
    "tests/test_radar_laser_processing.py::"
    "test_radar_recovers_independent_exact_velocity_at_native_rate"
)


def receipt(
    subfamilies: tuple[str, ...],
    dimension: str,
    rid: str | None = None,
    **changes: object,
) -> Receipt:
    base = {
        "id": rid or f"FX-{dimension}-{abs(hash(subfamilies)) % 10**6}",
        "dimension": dimension,
        "subfamilies": subfamilies,
        "claim": (
            "Fixture receipt: the engine output equals an independent closed form within tolerance."
        ),
        "issue": "RES-692",
        "implementation": (IMPLEMENTATION,),
        "reference": (REFERENCE,),
        "reference_kind": "closed_form",
        "independence": (
            "The closed form lives in the reference substrate, which imports no engine code."
        ),
        "tests": (TEST,),
        "tolerance": 1e-6,
        "tolerance_unit": "m/s",
        "tolerance_basis": "fixture tolerance chosen only for gate-logic tests",
    }
    base.update(changes)
    r = Receipt(**base)  # type: ignore[arg-type]
    digests = {
        ref: receipts_mod.resolve_digest(ROOT, ref)
        for ref in (*r.implementation, *r.reference, *r.tests)
    }
    return Receipt(**{**r.__dict__, "digests": {k: v for k, v in digests.items() if v}})


def fixture_receipts(model: Ledger) -> tuple[Receipt, ...]:
    return tuple(
        receipt((s.id,), d, rid=f"FX-{s.id}-{d}") for s in model.subfamilies for d in M3_DIMENSIONS
    )


def fixture_empirical(model: Ledger):  # type: ignore[no-untyped-def]
    """Structurally valid *measured* records, for gate-logic tests only (not real evidence)."""
    from dynamisbench_soccer.qualification.empirical import EmpiricalRecord

    return tuple(
        EmpiricalRecord(
            id=f"EX-{s.id}-{d}",
            subfamily=s.id,
            dimension=d,
            instrument="Fixture instrument description for gate-logic tests only",
            estimand="fixture estimand for gate-logic tests only",
            protocol="fixture protocol for gate-logic tests only",
            conditions="fixture conditions for gate-logic tests only",
            criterion_kind="paired_instrument" if d == "L" else "reference_standard",
            criterion="fixture criterion for gate-logic tests only",
            uncertainty="fixture uncertainty statement for gate-logic tests only",
            data_category="measured",
            data_sha256="b" * 64,
            rights_ref="README.md",
            n_independent_units=20,
            issue="RES-692",
        )
        for s in model.subfamilies
        for d in M3_DIMENSIONS
    )
