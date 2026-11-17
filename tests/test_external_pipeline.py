"""RES-720 gates 4-5: strict v2 conversion, source QC parity, diagnostics, end-to-end run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _external_synth import (
    MSS,
    REF,
    TAU,
    TD,
    csv_bytes,
    dump,
    rights_payload,
    sprint,
    upstream,
    verified_payload,
)

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.external import diagnostics, pipeline, shorts_laser, sourceqc
from dynamisbench_soccer.external.models import SignalStage
from dynamisbench_soccer.qc import spec
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import CANONICAL_UNIT, DeviceClass, Quantity

SESSION = shorts_laser.SessionEvidence("2024-01-01", "synthetic test evidence reference")


def _table(tmp_path: Path, cols: dict[str, np.ndarray] | None = None) -> shorts_laser.SourceTable:
    data = csv_bytes(cols if cols is not None else sprint())
    path = tmp_path / "laser.csv"
    path.write_bytes(data)
    return shorts_laser.parse(path, hashlib.sha256(data).hexdigest())


@pytest.fixture
def rules(repo_root: Path) -> spec.RuleSet:
    return spec.load(repo_root / "qc/rules.toml")


# --- schema-v2 conversion -----------------------------------------------------------------


def test_missing_session_date_blocks_conversion(tmp_path: Path) -> None:
    outcome = shorts_laser.convert(_table(tmp_path), None)
    assert not outcome.converted and outcome.bundle is None
    assert len(outcome.blockers) == 1 and outcome.blockers[0].startswith("session.date")
    assert dict(outcome.determinations)["test_id=sprint_fv"].startswith("protocol fit only")


@pytest.mark.parametrize(("date", "ref"), [("2024-13-01", "valid reference"), ("2024-01-01", "")])
def test_session_evidence_requires_real_date_and_reference(date: str, ref: str) -> None:
    with pytest.raises(ValueError):
        shorts_laser.SessionEvidence(date, ref)


def test_non_increasing_time_blocks_conversion(tmp_path: Path) -> None:
    cols = sprint(n=300)
    cols["time"] = cols["time"].copy()
    cols["time"][100] = cols["time"][99]
    outcome = shorts_laser.convert(_table(tmp_path, cols), SESSION)
    assert not outcome.converted and any(b.startswith("time:") for b in outcome.blockers)


def test_conversion_is_strict_and_truthful(tmp_path: Path) -> None:
    table = _table(tmp_path)
    bundle = shorts_laser.convert(table, SESSION).bundle
    assert bundle is not None
    m = bundle.meta
    assert m.test_id == "sprint_fv" and m.data_category == "empirical"
    assert "non-soccer" in m.protocol.protocol_id and m.protocol.start_mode == "standing"
    assert m.session.date == SESSION.date
    assert [c.name for c in m.channels] == [
        "time",
        "distance",
        "velocity",
        "velocity_vendor_unsmoothed",
    ]
    for c in m.channels:
        assert c.unit == CANONICAL_UNIT[c.quantity]
        assert c.description.startswith(f"[{SignalStage.MANUFACTURER_PROCESSED}]")
        assert "raw" not in c.name
    assert m.channel("time").quantity is Quantity.TIME
    # no calibration, geometry or clock knowledge is fabricated
    assert m.calibration == [] and m.acquisition is not None
    src = m.acquisition.sources[0]
    assert src.device.device_class is DeviceClass.LASER and src.calibration == []
    assert src.calibration_unknown_reason
    assert set(src.geometry.unknown) == {"mounting_height_m", "aiming_point"}
    assert src.geometry.mounting_height_m is None and src.geometry.aiming_point is None
    assert m.acquisition.clocks[0].source == "unknown"
    assert m.subject.body_mass_kg is None and m.subject.birth_year is None
    assert m.sampling is not None and m.sampling.nominal_hz == 1000.0
    assert m.provenance.source_sha256 == table.sha256
    assert any(t.startswith("NOT applied (package_derived)") for t in m.provenance.transforms)
    assert SESSION.evidence_ref in " ".join(m.provenance.transforms)
    # values are the source values, untouched
    for col in shorts_laser.SOURCE_COLUMNS:
        assert np.array_equal(bundle[col.channel], table.columns[col.source_column])


def test_converted_bundle_round_trips_through_v2_reader(tmp_path: Path) -> None:
    bundle = shorts_laser.convert(_table(tmp_path), SESSION).bundle
    assert bundle is not None
    stem = tmp_path / "out" / bundle.meta.bundle_id
    io.write(bundle, stem)
    back = io.read(stem)
    assert back.meta == bundle.meta
    assert sum(len(b["time"]) for b in io.iter_batches(stem, batch_rows=700)) == bundle.meta.n_rows


# --- QC -------------------------------------------------------------------------------------


def test_source_qc_matches_bundle_qc_evaluators(tmp_path: Path, rules: spec.RuleSet) -> None:
    table = _table(tmp_path)
    bundle = shorts_laser.convert(table, SESSION).bundle
    assert bundle is not None
    by_rule = {r.rule_id: r for r in spec.run(bundle, rules)}
    source = {(c.rule_id, c.name): c for c in sourceqc.run(table, rules)}
    for rid in ("QC-S03", "QC-S04", "QC-R01"):
        mirrored = next(c for (r, _), c in source.items() if r == rid)
        assert mirrored.status is by_rule[rid].status, rid
    r01 = source[("QC-R01", "radar_start_captured")]
    assert r01.values["start_speed"] == pytest.approx(by_rule["QC-R01"].values["start_speed"])
    r02 = source[("QC-R02", "radar_acceleration_spikes[velocity]")]
    assert r02.status is by_rule["QC-R02"].status
    assert r02.values["max_accel"] == pytest.approx(by_rule["QC-R02"].values["max_accel"])


def test_source_qc_detects_defects(tmp_path: Path, rules: spec.RuleSet) -> None:
    cols = sprint(n=1000)
    cols["time"] = cols["time"].copy()
    cols["time"][500:] += 0.010  # 10 ms gap
    cols["distance"] = cols["distance"].copy()
    cols["distance"][800] -= 0.5
    cols["velocity"] = cols["velocity"].copy()
    cols["velocity"][-10:] = cols["velocity"].max() + 1  # clipped run at max
    checks = {(c.rule_id, c.name): c for c in sourceqc.run(_table(tmp_path, cols), rules)}
    assert checks[("QC-S04", "timestamp_integrity")].status is spec.Status.FAIL
    assert checks[("SRC-02", "grid_matches_published_rate")].status is spec.Status.FAIL
    assert checks[("SRC-03", "distance_non_decreasing")].status is spec.Status.FAIL
    assert checks[("QC-S03", "clipping")].status is spec.Status.FAIL
    assert checks[("QC-R01", "radar_start_captured")].status is spec.Status.FAIL  # non-uniform
    assert sourceqc.verdict(list(checks.values())) == "reject"


# --- diagnostics ----------------------------------------------------------------------------


def test_fit_recovers_analytical_parameters() -> None:
    cols = sprint(n=6000)
    fit = diagnostics.fit_mono_exponential(cols["time"], cols["velocity"])
    assert fit.mss_m_s == pytest.approx(MSS, abs=1e-4)
    assert fit.tau_s == pytest.approx(TAU, abs=1e-4)
    assert fit.td_s == pytest.approx(TD, abs=1e-4)
    assert fit.rmse_m_s < 1e-4


def test_diagnostics_on_consistent_synthetic_source(tmp_path: Path) -> None:
    d = diagnostics.compute(_table(tmp_path, sprint(n=6000)))
    tb: Any = d["distance_velocity_time_base"]
    assert tb["best_lag_ms"] == pytest.approx(0, abs=1.5)
    assert tb["rms_at_zero_lag_m_s"] < 0.02
    integ: Any = d["integrated_velocity_vs_distance"]
    assert abs(integ["velocity"]["end_difference_m"]) < 0.002
    rows: Any = d["distance_derivative_vs_vendor_velocity"]
    # residual is bounded by the analytical quantisation floor, not hidden by smoothing
    assert rows[-1]["vs_velocity"]["rms"] < 3 * rows[-1]["quantisation_floor_rms_m_s"] + 1e-3
    fits: Any = d["mono_exponential_fit_sensitivity"]
    assert fits["variants"]["velocity_all_rows"]["mss_m_s"] == pytest.approx(MSS, abs=1e-3)


def test_diagnostics_detect_a_time_base_offset(tmp_path: Path) -> None:
    cols = sprint(n=6000)
    shifted = sprint(n=6000, td=TD - 0.5)
    cols["distance"] = shifted["distance"]  # distance leads velocity by 500 ms
    tb: Any = diagnostics.compute(_table(tmp_path, cols))["distance_velocity_time_base"]
    assert tb["best_lag_ms"] == pytest.approx(500, abs=2)
    assert tb["rms_at_best_lag_m_s"] < 0.05 < tb["rms_at_zero_lag_m_s"]


# --- end to end -----------------------------------------------------------------------------


def _setup(tmp_path: Path, rights: dict[str, object]) -> tuple[Path, Path, str]:
    up = tmp_path / "up"
    lock = upstream(up, csv_bytes(sprint(n=6000)))
    ext = tmp_path / "external"
    dump(lock.model_dump(mode="json"), ext / lock.source_id / "source-lock.json")
    dump(rights, ext / lock.source_id / "rights.json")
    return up, ext, lock.file(shorts_laser.DATA_PATH).sha256


def _run(tmp_path: Path, repo_root: Path, rights: dict[str, object], **kw: Any) -> dict[str, Any]:
    up, ext, sha = _setup(tmp_path, rights)
    return pipeline.run_shorts_laser(
        tmp_path / "run",
        external_root=ext,
        rules_path=repo_root / "qc/rules.toml",
        source_dir=up,
        quarantine=tmp_path / "q",
        expected_ref=kw.pop("expected_ref", REF),
        expected_sha256=sha,
        source_id="synthetic-laser",
        store_root=tmp_path / "store",
        staging=tmp_path / "stage",
        **kw,
    )


def _status(report: dict[str, Any]) -> dict[str, str]:
    return {s["name"]: s["status"] for s in report["stages"]}


def test_pending_rights_run_quarantine_and_block_honestly(tmp_path: Path, repo_root: Path) -> None:
    report = _run(tmp_path, repo_root, rights_payload())
    assert _status(report) == {
        "pinned_source": "passed",
        "integrity": "passed",
        "rights_verification": "blocked",
        "ingestion": "blocked",
        "safe_parse": "passed",
        "profile": "passed",
        "schema_v2_conversion": "blocked",
        "qc": "passed",
        "diagnostics": "passed",
    }
    assert report["outcome"].startswith("completed_with_blocked_gates")
    assert not (tmp_path / "store").exists() and not (tmp_path / "stage").exists()
    first = (tmp_path / "run" / "report.json").read_bytes()
    _run(tmp_path, repo_root, rights_payload())
    assert (tmp_path / "run" / "report.json").read_bytes() == first  # deterministic
    assert (tmp_path / "run" / "report.md").read_text().startswith("# External-source")


def test_verified_rights_and_date_complete_canonical_path(tmp_path: Path, repo_root: Path) -> None:
    report = _run(tmp_path, repo_root, verified_payload(), session=SESSION)
    assert report["outcome"] == "completed"
    assert set(_status(report).values()) == {"passed"}
    assert report["conversion"]["bundle"]["rows_streamed"] == 6000
    assert report["bundle_qc"]["verdict"] in {"accept", "accept_with_flags"}
    assert any((tmp_path / "store").rglob("*"))


def test_tampered_source_fails_closed(tmp_path: Path, repo_root: Path) -> None:
    up, ext, sha = _setup(tmp_path, rights_payload())
    csv = up / shorts_laser.DATA_PATH
    csv.write_bytes(csv.read_bytes().replace(b"\n0.100,", b"\n0.101,", 1))
    report = pipeline.run_shorts_laser(
        tmp_path / "run",
        external_root=ext,
        rules_path=repo_root / "qc/rules.toml",
        source_dir=up,
        quarantine=tmp_path / "q",
        expected_ref=REF,
        expected_sha256=sha,
        source_id="synthetic-laser",
    )
    assert report["outcome"] == "failed" and _status(report)["integrity"] == "failed"
    assert "diagnostics" not in report and "profile" not in report


def test_unverifiable_license_evidence_fails(tmp_path: Path, repo_root: Path) -> None:
    report = _run(tmp_path, repo_root, rights_payload(copyright_holder="Nobody"))
    assert _status(report)["rights_verification"] == "failed" and "profile" not in report


def test_no_local_inspection_stops_before_reading_data(tmp_path: Path, repo_root: Path) -> None:
    report = _run(tmp_path, repo_root, rights_payload(local_inspection_permitted=False))
    assert _status(report)["ingestion"] == "blocked"
    assert report["outcome"] == "stopped_at_blocked_gate" and "profile" not in report


def test_adapter_version_binding(tmp_path: Path, repo_root: Path) -> None:
    report = _run(tmp_path, repo_root, rights_payload(), expected_ref="a" * 40)
    assert _status(report) == {"pinned_source": "failed"}


def test_cli_requires_date_evidence(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["external", "run", "shorts-laser", "--session-date", "2024-01-01"]) == 1
    assert "needs --session-date-evidence" in capsys.readouterr().err


# --- the committed real-source report -------------------------------------------------------


def _longest_list(x: Any) -> int:
    if isinstance(x, dict):
        return max((_longest_list(v) for v in x.values()), default=0)
    if isinstance(x, list):
        return max([len(x), *(_longest_list(v) for v in x)])
    return 0


def test_committed_report_is_aggregate_and_honest(repo_root: Path) -> None:
    base = repo_root / "docs/external-data/reports/shorts-laser-diagnostic-report"
    report = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["source"]["data_sha256"] == shorts_laser.PINNED_SHA256
    assert report["source"]["pinned_ref"] == shorts_laser.PINNED_REF
    status = _status(report)
    assert status["rights_verification"] == status["ingestion"] == "blocked"
    assert status["schema_v2_conversion"] == "blocked" and status["diagnostics"] == "passed"
    assert not report["conversion"]["converted"] and "bundle_qc" not in report
    assert report["rights"]["redistribution_allowed"] is False
    assert report["profile"]["n_rows"] == 5232
    assert report["diagnostics"]["package_preprocessing"]["reproduces_documented_rows"]
    assert _longest_list(report) < 50  # no per-sample arrays are published
    md = base.with_suffix(".md").read_text(encoding="utf-8")
    assert "not reference truth" in md and "/home/" not in md
    assert pipeline.dumps(report) == base.with_suffix(".json").read_text(encoding="utf-8")
