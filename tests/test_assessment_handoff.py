"""RES-673: source-bound engine handoff, native F-V/COD pipelines and milestone integration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from test_gnss_lps_acquisition import track_case

from dynamisbench_soccer.assessment import handoff
from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines import gnss_lps
from dynamisbench_soccer.engines.gnss_lps_cli import _json_value
from dynamisbench_soccer.engines.radar_laser import ProcessingOptions, process
from dynamisbench_soccer.engines.radar_laser_acquisition import ChannelStage
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.motion_models import MonoExponentialSprint

TRUTH = MonoExponentialSprint(9.0, 1.2)
MASS = 80.0


def _radar_like(kind: str, tmp_path: Path, stage: str = "native") -> Path:
    case = export_case(f"radar_laser.{kind}", hz=100)
    options = ProcessingOptions(
        channel_stage=ChannelStage.NATIVE,
        correction_method="declared_fixed_angle" if kind == "radar" else "none",
    )
    report = json.loads(json.dumps(dataclasses.asdict(process(case.observation, options))))
    report["source_stage"] = stage
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(report))
    return path


def _track(kind: str, tmp_path: Path) -> Path:
    case, declaration = track_case(kind)
    result = gnss_lps.process(case.observation, declaration)
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(dataclasses.asdict(result), default=_json_value))
    return path


def _fv_config(tmp_path: Path, source: str, **extra: object) -> Path:
    path = tmp_path / f"fv_{source}.json"
    path.write_text(json.dumps({"source": source, "fv": {"mass_kg": MASS}, **extra}))
    return path


# ---- typed adapters -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["radar", "laser"])
def test_native_radar_and_laser_reports_load_with_identity(kind: str, tmp_path: Path) -> None:
    h = handoff.load_speed_report(_radar_like(kind, tmp_path), declared_source=kind)
    assert h.source == kind and h.stage == "native" and len(h.time_s) > 100
    assert h.provenance["start_method"] == "record_origin"
    assert h.provenance["time_origin"]


@pytest.mark.parametrize(
    ("kind", "source"),
    [("gnss", "gnss_doppler"), ("gnss", "gnss_position"), ("lps", "lps_position")],
)
def test_native_gnss_and_lps_reports_load_through_their_actual_schema(
    kind: str, source: str, tmp_path: Path
) -> None:
    h = handoff.load_speed_report(_track(kind, tmp_path), declared_source=source)
    assert h.source == source and h.device == kind and h.stage == "native"
    assert h.provenance["source_clock"] and h.provenance["reference_clock"]
    assert h.provenance["alignment_uncertainty_s"] is not None
    assert float(h.speed_m_s.max()) > 0


def test_doppler_and_positional_speed_are_different_traces(tmp_path: Path) -> None:
    path = _track("gnss", tmp_path)
    a = handoff.load_speed_report(path, declared_source="gnss_doppler")
    b = handoff.load_speed_report(path, declared_source="gnss_position")
    assert a.estimand != b.estimand and not np.array_equal(a.speed_m_s, b.speed_m_s)


@pytest.mark.parametrize(
    ("report_kind", "wrong"),
    [("radar", "gnss_doppler"), ("radar", "laser"), ("laser", "radar"), ("radar", "lps_position")],
)
def test_radar_trace_with_a_wrong_source_label_is_refused(
    report_kind: str, wrong: str, tmp_path: Path
) -> None:
    with pytest.raises(handoff.HandoffRefusal):
        handoff.load_speed_report(_radar_like(report_kind, tmp_path), declared_source=wrong)


@pytest.mark.parametrize(
    ("kind", "wrong"),
    [
        ("gnss", "lps_position"),
        ("lps", "gnss_position"),
        ("lps", "gnss_doppler"),
        ("gnss", "radar"),
    ],
)
def test_track_with_a_wrong_source_label_is_refused(kind: str, wrong: str, tmp_path: Path) -> None:
    with pytest.raises(handoff.HandoffRefusal):
        handoff.load_speed_report(_track(kind, tmp_path), declared_source=wrong)


def test_vendor_processed_stage_needs_explicit_opt_in(tmp_path: Path) -> None:
    path = _radar_like("radar", tmp_path, stage="manufacturer_processed")
    with pytest.raises(handoff.HandoffRefusal, match="not native"):
        handoff.load_speed_report(path, declared_source="radar")
    allowed = handoff.load_speed_report(path, declared_source="radar", allow_vendor_processed=True)
    assert allowed.stage == "manufacturer_processed"
    # the engine report is not mutated into "native"
    assert json.loads(path.read_text())["source_stage"] == "manufacturer_processed"


@pytest.mark.parametrize(
    "flag", [{"verdict": "reject"}, {"qc_verdict": "failed"}, {"diagnostic_only": True}]
)
def test_rejected_or_diagnostic_reports_never_feed_accepted_metrics(
    flag: dict[str, object], tmp_path: Path
) -> None:
    path = _radar_like("radar", tmp_path)
    report = {**json.loads(path.read_text()), **flag}
    path.write_text(json.dumps(report))
    with pytest.raises(handoff.HandoffRefusal):
        handoff.load_speed_report(path, declared_source="radar")
    # negative control: the same report without the flag is accepted
    del_keys = {**report}
    for key in flag:
        del_keys.pop(key)
    path.write_text(json.dumps(del_keys))
    assert handoff.load_speed_report(path, declared_source="radar").source == "radar"


def _mutated(path: Path, **changes: object) -> Path:
    report = {**json.loads(path.read_text()), **changes}
    path.write_text(json.dumps(report))
    return path


def test_negative_speed_is_refused_not_absolute_valued(tmp_path: Path) -> None:
    path = _radar_like("radar", tmp_path)
    report = json.loads(path.read_text())
    speeds = report["velocity_m_s"]
    reversed_trace = [-x for x in speeds]
    _mutated(path, velocity_m_s=reversed_trace)
    with pytest.raises(handoff.HandoffRefusal, match="negative speed"):
        handoff.load_speed_report(path, declared_source="radar")


@pytest.mark.parametrize(
    "change",
    [
        {"velocity_m_s": [0.0, 1.0]},  # length mismatch and too short
        {"time_from_start_s": "abc"},
        {"native_quantity": "mystery"},
    ],
)
def test_malformed_reports_are_refused(change: dict[str, object], tmp_path: Path) -> None:
    path = _mutated(_radar_like("radar", tmp_path), **change)
    with pytest.raises(handoff.HandoffRefusal):
        handoff.load_speed_report(path, declared_source="radar")


def test_nonmonotonic_nonfinite_time_is_refused(tmp_path: Path) -> None:
    path = _radar_like("radar", tmp_path)
    t = json.loads(path.read_text())["time_from_start_s"]
    swapped = [*t]
    swapped[10], swapped[11] = swapped[11], swapped[10]
    _mutated(path, time_from_start_s=swapped)
    with pytest.raises(handoff.HandoffRefusal, match="increasing"):
        handoff.load_speed_report(path, declared_source="radar")


def test_unrecognised_report_and_bad_json_are_refused(tmp_path: Path) -> None:
    other = tmp_path / "x.json"
    other.write_text(json.dumps({"hello": 1}))
    with pytest.raises(handoff.HandoffRefusal):
        handoff.load_speed_report(other, declared_source="radar")
    other.write_text("not json")
    with pytest.raises(handoff.HandoffRefusal):
        handoff.load_speed_report(other, declared_source="radar")


# ---- CLI pipelines ----------------------------------------------------------------------------


def _run(cmd: str, path: Path, config: Path, out: Path) -> tuple[int, dict[str, object]]:
    code = main(["assess", cmd, str(path), "--config", str(config), "--out", str(out)])
    return code, json.loads(out.read_text())


def test_native_radar_and_laser_reach_sprint_fv_with_input_provenance(tmp_path: Path) -> None:
    for kind in ("radar", "laser"):
        path = _radar_like(kind, tmp_path)
        config = _fv_config(tmp_path, kind, milestone_distances_m=[10, 20, 30])
        code, report = _run("sprint-fv", path, config, tmp_path / f"{kind}_fv.json")
        assert code == 0 and report["fit"]["identifiable"] is True  # type: ignore[index]
        assert report["input"]["source"] == kind and report["input"]["stage"] == "native"  # type: ignore[index]
        marks = report["milestones_s"]
        for d in (10, 20, 30):
            assert marks[f"{d}_m"] == pytest.approx(TRUTH.gate_time_s(d), abs=0.02)  # type: ignore[index]


def test_native_gnss_and_lps_reach_sprint_fv_or_a_scientific_refusal(tmp_path: Path) -> None:
    """A short synthetic track may be unidentifiable; it must refuse, never fabricate."""
    for kind, source in (
        ("gnss", "gnss_doppler"),
        ("gnss", "gnss_position"),
        ("lps", "lps_position"),
    ):
        path = _track(kind, tmp_path)
        code, report = _run(
            "sprint-fv", path, _fv_config(tmp_path, source), tmp_path / f"{source}.json"
        )
        assert code in {0, 1}
        assert report["input"]["source"] == source  # type: ignore[index]
        assert report["input"]["provenance"]["alignment_uncertainty_s"] is not None  # type: ignore[index]
        if code == 1:
            assert report["fit"]["refusals"] and report["fit"]["f0_n"] is None  # type: ignore[index]


def test_cli_refuses_mislabelled_and_vendor_inputs(tmp_path: Path) -> None:
    radar = _radar_like("radar", tmp_path)
    out = tmp_path / "o.json"
    for bad in ("gnss_doppler", "lps_position", "laser"):
        code, report = _run("sprint-fv", radar, _fv_config(tmp_path, bad), out)
        assert code == 2 and "error" in report
    vendor = _radar_like("laser", tmp_path, stage="manufacturer_processed")
    code, report = _run("sprint-fv", vendor, _fv_config(tmp_path, "laser"), out)
    assert code == 2 and "not native" in report["error"]  # type: ignore[operator]
    code, report = _run(
        "sprint-fv", vendor, _fv_config(tmp_path, "laser", allow_vendor_processed=True), out
    )
    assert code == 0 and report["input"]["stage"] == "manufacturer_processed"  # type: ignore[index]


def test_cod_source_must_match_the_report(tmp_path: Path) -> None:
    radar = _radar_like("radar", tmp_path)
    cod_cfg = tmp_path / "cod.json"
    spec = {
        "declared_path": "straight 30 m",
        "path_length_m": 30,
        "turn_distance_m": 20,
        "turn_angle_deg": 45,
    }
    cod_cfg.write_text(json.dumps({"cod": {**spec, "source": "lps_position"}}))
    code, report = _run("cod", radar, cod_cfg, tmp_path / "cod_out.json")
    assert code == 2 and "identifies" in report["error"]  # type: ignore[operator]
    cod_cfg.write_text(json.dumps({"cod": {**spec, "source": "radar"}}))
    code, report = _run("cod", radar, cod_cfg, tmp_path / "cod_ok.json")
    assert code == 0 and report["input"]["source"] == "radar"  # type: ignore[index]


def test_modules_keep_the_reference_substrate_out() -> None:
    import ast

    text = Path(handoff.__file__).read_text()
    imported = {
        n.module for n in ast.walk(ast.parse(text)) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not any("reference" in m for m in imported)


def test_sub_tolerance_negative_rest_noise_is_clamped_explicitly_and_counted(
    tmp_path: Path,
) -> None:
    path = _radar_like("radar", tmp_path)
    report = json.loads(path.read_text())
    report["velocity_m_s"][0] = -0.01
    path.write_text(json.dumps(report))
    h = handoff.load_speed_report(path, declared_source="radar")
    assert (
        float(h.speed_m_s.min()) == 0.0
        and h.input_record()["clamped_sub_tolerance_negative_samples"] == 1
    )
    out = tmp_path / "cod_out.json"
    cfg = tmp_path / "cod.json"
    cfg.write_text(
        json.dumps(
            {
                "cod": {
                    "declared_path": "straight 30 m",
                    "path_length_m": 30,
                    "turn_distance_m": 20,
                    "turn_angle_deg": 45,
                    "source": "radar",
                }
            }
        )
    )
    code, rep = _run("cod", path, cfg, out)
    assert code == 0 and rep["input"]["clamped_sub_tolerance_negative_samples"] == 1  # type: ignore[index]


def test_gnss_doppler_basis_and_timing_provenance_are_enforced(tmp_path: Path) -> None:
    path = _track("gnss", tmp_path)
    h = handoff.load_speed_report(path, declared_source="gnss_doppler")
    assert h.provenance["doppler_basis"] == "horizontal" and h.provenance["latency_s"] is not None
    assert "horizontal" in h.estimand
    report = json.loads(path.read_text())
    report["acquisition"]["declaration"]["doppler_basis"] = "three_dimensional"
    path.write_text(json.dumps(report))
    with pytest.raises(handoff.HandoffRefusal, match="not horizontal"):
        handoff.load_speed_report(path, declared_source="gnss_doppler")
    # the positional estimand of the same report is unaffected by the Doppler basis
    assert (
        handoff.load_speed_report(path, declared_source="gnss_position").source == "gnss_position"
    )
