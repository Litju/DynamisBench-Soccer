from pathlib import Path

import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.signals import examples, io


def test_json_schema_doc_in_sync(repo_root: Path) -> None:
    doc = repo_root / "docs" / "schemas" / "signal-bundle-meta.v2.json"
    assert doc.read_text(encoding="utf-8") == io.json_schema()


def test_cli_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    io.write(examples.cmj_dual_plate(), tmp_path / "cmj")
    assert main(["signals", "inspect", str(tmp_path / "cmj.parquet")]) == 0
    out = capsys.readouterr().out
    assert "fz_left" in out and "vendor_reported (not reference values)" in out
