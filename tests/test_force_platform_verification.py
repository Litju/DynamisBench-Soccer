"""Reproducible numerical evidence: run this file with --write to refresh tables."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from dynamisbench_soccer.engines.force_platform import MethodSpec, process
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import cmj, dj, hops, sj

ROOT = Path(__file__).resolve().parents[1]


def verification_rows() -> list[dict[str, str]]:
    rows = []
    variants = {
        "baseline": {},
        "butterworth": {"filter": "butterworth"},
        "chebyshev": {"filter": "chebyshev"},
        "simpson": {"integration": "simpson"},
        "reverse_scan": {"onset": "reverse_scan"},
        "first_derivative": {"onset": "first_derivative"},
        "absolute_N": {"onset": "absolute_N"},
        "weight_drop": {"contact": "system_weight_drop"},
        "unloaded_sd": {"contact": "k_sd_unloaded"},
        "start_onset": {"integration_start": "onset"},
        "start_record": {"integration_start": "record_start"},
    }
    for factory in (cmj, sj, dj, hops):
        motion = factory()
        external = factory in (dj, hops)
        for family in ("force_platform.single", "force_platform.dual"):
            for rate in (600, 1000, 2000):
                observation = export_case(family, motion, hz=rate).observation
                weighing = export_case(family, cmj(), hz=rate).observation if external else None
                for name, changes in variants.items():
                    if external and name == "start_onset":
                        continue  # No preceding quiet onset was recorded.
                    settings = {
                        "unloaded_window_s": (
                            float(motion.takeoff_s() + 0.1),
                            float(motion.takeoff_s() + 0.2),
                        ),
                        "integration_start": "record_start" if external else "quiet_start",
                        "initial_velocity_m_s": motion.initial_velocity_m_s if external else None,
                        **changes,
                    }
                    result = process(
                        observation, method=MethodSpec.model_validate(settings), weighing=weighing
                    )
                    truth_v = np.array([motion.state(float(t)).velocity_m_s for t in result.time_s])
                    truth_x = np.array(
                        [motion.state(float(t)).displacement_m for t in result.time_s]
                    )
                    valid = np.isfinite(result.velocity_m_s)
                    v_error = float(np.max(np.abs(result.velocity_m_s[valid] - truth_v[valid])))
                    x_error = float(np.max(np.abs(result.displacement_m[valid] - truth_x[valid])))
                    assert v_error < 0.16 and x_error < 0.1
                    flight = result.flights[0]
                    assert abs(flight.takeoff_s - motion.takeoff_s()) <= 1 / rate + 1e-9
                    rows.append(
                        {
                            "scenario": str(motion.assessment_id),
                            "subfamily": family,
                            "hz": str(rate),
                            "variant": name,
                            "onset_s": ""
                            if result.onset.time_s is None
                            else f"{result.onset.time_s:.6f}",
                            "takeoff_s": f"{flight.takeoff_s:.6f}",
                            "landing_s": f"{flight.landing_s:.6f}",
                            "takeoff_velocity_m_s": f"{flight.takeoff_velocity_m_s:.6f}",
                            "height_velocity_m": f"{flight.height_velocity_m:.6f}",
                            "height_flight_m": f"{flight.height_flight_time_m:.6f}",
                            "max_velocity_error_m_s": f"{v_error:.6f}",
                            "max_displacement_error_m": f"{x_error:.6f}",
                            "qc_verdict": result.verdict,
                        }
                    )
    return rows


def evidence_text(rows: list[dict[str, str]]) -> tuple[str, str]:
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    summary = [
        "# RES-670 numerical verification",
        "",
        (
            "Generated from `tests/test_force_platform_verification.py`; no empiric"
            "al device validity is asserted."
        ),
        "",
        "| Scenario | Filter | Max velocity error (m/s) | Max displacement error (m) |",
        "| -- | -- | -- | -- |",
    ]
    for scenario in ("cmj", "sj", "dj", "rj_10_5"):
        for variant in ("baseline", "butterworth", "chebyshev"):
            matching = [r for r in rows if r["scenario"] == scenario and r["variant"] == variant]
            v = max(float(r["max_velocity_error_m_s"]) for r in matching)
            x = max(float(r["max_displacement_error_m"]) for r in matching)
            summary.append(f"| {scenario} | {variant} | {v:.6f} | {x:.6f} |")
    summary += [
        "",
        f"{len(rows)} rows cover both plate subfamilies at 600, 1000 and 2000 Hz.",
        (
            "Full method/onset/contact/start sensitivity values: [force-platform-ve"
            "rification.csv](force-platform-verification.csv)."
        ),
        (
            "Baseline uses raw force and trapezoid integration. Filtered results in"
            "clude boundary smoothing bias."
        ),
        (
            "The separate exhaustive tests cross all filter/integration/onset choic"
            "es on every scenario and rate."
        ),
        (
            "Ideal flat analytical plateaus trigger the pre-existing QC-S03 clippin"
            "g heuristic; these are diagnostic numerical results, not accepted empi"
            "rical trials."
        ),
        (
            "No pre-movement onset exists in the recorded DJ/hop fixtures; those ro"
            "ws use separate quiet weighing and explicit initial velocity. An onset"
            "-start reconstruction is inapplicable there."
        ),
        (
            "All reported errors are against independent RES-669 closed-form trajec"
            "tories, not vendor outputs or values calculated by the production engi"
            "ne."
        ),
        (
            "These conditional sensitivity tables are not confidence intervals or h"
            "ardware uncertainty budgets."
        ),
    ]
    return table.getvalue(), "\n".join(summary) + "\n"


def test_numeric_verification_and_sensitivity_evidence_are_current() -> None:
    table, summary = evidence_text(verification_rows())
    assert (ROOT / "docs/force-platform-verification.csv").read_text() == table
    assert (ROOT / "docs/force-platform-verification.md").read_text() == summary


if __name__ == "__main__":
    import sys

    assert sys.argv[1:] == ["--write"], "use --write to regenerate numerical evidence"
    table, summary = evidence_text(verification_rows())
    (ROOT / "docs/force-platform-verification.csv").write_text(table)
    (ROOT / "docs/force-platform-verification.md").write_text(summary)
