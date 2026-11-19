"""Real file acquisition → independently checked metrics → bound output traces."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.reference.kinematics import G
from dynamisbench_soccer.signals import io


def csv_record(path: Path, dual: bool) -> None:
    pair = export_case("force_platform.dual" if dual else "force_platform.single", hz=1000)
    names = tuple(n for n in pair.observation.data if n != "time")
    noisy = inject(
        pair.observation, ContinuousFault(ContinuousKind.NOISE, names, magnitude=0.1, seed=7)
    )
    headers = """# dbs-generic-csv: 1
# data_category: synthetic
# subject: BENCH-001
# session_id: REF-670
# date: 2000-01-01
# test_id: cmj
# protocol_id: cmj.hands_on_hips.v1
# arm_swing: hands_on_hips
# sampling_rate_hz: 1000
# zeroing: yes
# zeroing_method: declared upstream unloaded zero
"""
    columns = {"time[s]": noisy["time"], **{name + "[N]": noisy[name] for name in names}}
    with path.open("w") as stream:
        stream.write(headers)
        np.savetxt(
            stream,
            np.column_stack(list(columns.values())),
            delimiter=",",
            header=",".join(columns),
            comments="",
        )


@pytest.mark.parametrize("dual", [False, True])
def test_generic_csv_to_actual_metric_and_trace_files(tmp_path: Path, dual: bool) -> None:
    source = tmp_path / "acquisition.csv"
    csv_record(source, dual)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "method": {"assume_shared_clock": dual},
                "plates": [
                    {"plate": p, "assume_field_axes": True} for p in ((1, 2) if dual else (1,))
                ],
            }
        )
    )
    out = tmp_path / "metrics.json"
    traces = tmp_path / "traces.csv"
    assert (
        main(
            [
                "force-platform",
                "process",
                str(source),
                "--csv",
                "--config",
                str(config),
                "--out",
                str(out),
                "--traces",
                str(traces),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert result["verdict"] != "reject"
    assert result["mass_kg"] == pytest.approx(75, abs=0.005)
    assert abs(result["flights"][0]["takeoff_velocity_m_s"] - 2.7) < 0.02
    assert abs(result["flights"][0]["height_velocity_m"] - 2.7**2 / (2 * G)) < 0.005
    assert result["trace_output"]["sha256"] == hashlib.sha256(traces.read_bytes()).hexdigest()
    assert (
        result["method_spec"]["source_provenance"][0]["source_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    data = np.genfromtxt(traces, delimiter=",", names=True)
    assert len(data) > 2000
    assert data["time_s"][1] - data["time_s"][0] == pytest.approx(0.001)
    assert data["velocity_m_s"][1750] == pytest.approx(2.7, abs=0.02)


def test_verified_bundle_and_explicit_rejection_reports(tmp_path: Path) -> None:
    io.write(export_case("force_platform.single", hz=1000).observation, tmp_path / "bundle")
    # Ideal flat plateaus trigger existing clipping heuristic, without hiding computed diagnostics.
    out = tmp_path / "report.json"
    assert main(["force-platform", "process", str(tmp_path / "bundle"), "--out", str(out)]) == 1
    result = json.loads(out.read_text())
    assert result["diagnostic_only"]
    assert result["flights"][0]["takeoff_velocity_m_s"] == pytest.approx(2.7, abs=0.01)
    (tmp_path / "bad.json").write_text('{"method":{"not_a_method":true}}')
    assert (
        main(
            [
                "force-platform",
                "process",
                str(tmp_path / "bundle"),
                "--config",
                str(tmp_path / "bad.json"),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    assert "error" in json.loads(out.read_text())


def test_cli_preserves_acquisition_and_configuration_files(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    csv_record(source, False)
    before = source.read_bytes()
    assert main(["force-platform", "process", str(source), "--csv", "--out", str(source)]) == 2
    assert source.read_bytes() == before


def test_cli_malformed_qc_rules_return_controlled_json_rejection(tmp_path: Path) -> None:
    source = export_case("force_platform.single", hz=1000).observation
    io.write(source, tmp_path / "trial")
    rules = Path(__file__).resolve().parents[1] / "qc/rules.toml"
    bad = tmp_path / "bad-rules.toml"
    bad.write_text(rules.read_text().replace("params = { min_run = 5 }", "params = {}"))
    out = tmp_path / "failure.json"
    assert (
        main(
            [
                "force-platform",
                "process",
                str(tmp_path / "trial"),
                "--qc-rules",
                str(bad),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    report = json.loads(out.read_text())
    assert report["verdict"] == "reject"
    assert "QC parameter names" in report["error"]
