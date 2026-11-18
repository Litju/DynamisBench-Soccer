"""RES-699 installed-interface and checksum-gated CLI checks."""

from __future__ import annotations

import json
from pathlib import Path

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.reference.exporters import export_case


def test_radar_and_laser_cli_are_distinct(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    for kind in ("radar", "laser"):
        folder = tmp_path / kind
        export_case(f"radar_laser.{kind}", hz=100).write(folder)
        output = tmp_path / f"{kind}.json"
        args = [
            "radar-laser",
            "process",
            str(folder / "observation"),
            "--stage",
            "native",
            "--out",
            str(output),
        ]
        if kind == "radar":
            args.extend(["--correction", "declared_fixed_angle"])
        assert main(args) == 0
        result = json.loads(output.read_text())
        assert result["source_stage"] == "native"
        assert result["native_quantity"] == ("velocity" if kind == "radar" else "distance")
        assert result["peak_velocity_m_s"] > 0
    assert capsys.readouterr().err == ""


def test_cli_refuses_bad_cosine_and_tampering(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "laser"
    export_case("radar_laser.laser", hz=100).write(folder)
    args = ["radar-laser", "process", str(folder / "observation"), "--stage", "native"]
    assert main([*args, "--correction", "declared_fixed_angle"]) == 2
    assert "calibration missing" in capsys.readouterr().err
    meta = folder / "observation.json"
    raw = json.loads(meta.read_text())
    raw["n_rows"] += 1
    meta.write_text(json.dumps(raw))
    assert main(args) == 2
    assert "Parquet columns/row count" in capsys.readouterr().err
