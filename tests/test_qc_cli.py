from pathlib import Path

import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.qc import spec
from dynamisbench_soccer.signals import examples, io


@pytest.fixture(scope="module")
def repo_root_module() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rules(repo_root_module: Path) -> spec.RuleSet:
    return spec.load(repo_root_module / "qc" / "rules.toml")


def test_cli_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dynamisbench_soccer.qc import faults

    io.write(examples.cmj_dual_plate(), tmp_path / "clean")
    io.write(faults.double_contact(), tmp_path / "bad")
    assert main(["qc", "run", str(tmp_path / "clean")]) == 0
    assert main(["qc", "run", str(tmp_path / "bad")]) == 1
    assert "verdict: reject" in capsys.readouterr().out


def test_rules_doc_in_sync(repo_root_module: Path, rules: spec.RuleSet) -> None:
    from dynamisbench_soccer.qc.cli import rules_markdown

    doc = (repo_root_module / "docs" / "qc-rules.md").read_text(encoding="utf-8")
    assert doc == rules_markdown(rules)
