"""Frozen seeded artifacts and ledger honesty at the RES-669 acceptance boundary."""

import hashlib
import json
from pathlib import Path

from dynamisbench_soccer.qualification.ledger import (
    DIMENSIONS,
    SUBFAMILIES,
    load,
    validate,
)
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import PositionFault, PositionKind, inject
from dynamisbench_soccer.signals import io

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_seeded_v2_artifact_bytes(tmp_path: Path) -> None:
    expected = json.loads((ROOT / "tests/fixtures/reference/seeded-v1.json").read_text())
    case = export_case("gnss_lps.gnss", hz=10)
    fault = PositionFault(
        PositionKind.NOISE, index=0, count=case.observation.meta.n_rows, noise_sd_m=0.03, seed=17
    )
    corrupted = inject(case.observation, fault)
    case.write(tmp_path)
    io.write(corrupted, tmp_path / "corrupted")
    for name, digest in expected["sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest, name


def test_full_qualification_contract_and_actual_reference_bindings_remain_honest() -> None:
    ledger = load(ROOT / "qualification/ledger.toml")
    assert not validate(ledger, ROOT)
    assert {s.id for s in ledger.subfamilies} == SUBFAMILIES
    assert len(ledger.subfamilies) == 15
    assert len({s.family for s in ledger.subfamilies}) == 8
    for sub in ledger.subfamilies:
        assert set(sub.ratings) == set(DIMENSIONS)
        assert set(sub.owners) == set(DIMENSIONS)
        refs = [e for e in ledger.evidence if e.subfamily == sub.id and e.issue == "RES-669"]
        assert any(
            e.dimension == "J" and e.ref == "repo:src/dynamisbench_soccer/reference/exporters.py"
            for e in refs
        )
        for ref in (
            "tests/test_reference_exporters.py",
            "tests/test_reference_faults.py",
            "tests/test_reference_acceptance.py",
        ):
            assert any(e.dimension == "K" and e.ref == "repo:" + ref for e in refs)
