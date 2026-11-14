from pathlib import Path

import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.signals import io

FIX = Path(__file__).parent / "fixtures" / "adapters"


def test_cli_import_and_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "nhe"
    assert (
        main(
            [
                "adapters",
                "import",
                "generic_dynamometer_csv",
                str(FIX / "valid" / "nhe_kgf.csv"),
                str(out),
            ]
        )
        == 0
    )
    assert io.read(out).meta.test_id == "nhe"
    assert (
        main(
            [
                "adapters",
                "import",
                "generic_forceplate_csv",
                str(FIX / "invalid" / "fp_unknown_unit.csv"),
                str(tmp_path / "x"),
            ]
        )
        == 1
    )
    assert main(["adapters", "list"]) == 0
    assert "generic_forceplate_csv@0.1" in capsys.readouterr().out
