import runpy
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_biosyngen_force.py"
COMPARATOR = runpy.run_path(str(SCRIPT))


def test_comparison_uses_independent_momentum_and_keeps_processor_refusal():
    # A constant 20 N net over 0.5 s raises model momentum by 10 N s.
    traces = np.zeros(
        4,
        dtype=[
            ("time_s", "f8"),
            ("force_n", "f8"),
            ("fz_plate1_n", "f8"),
            ("fz_plate2_n", "f8"),
        ],
    )
    traces["time_s"] = [0, 0.25, 0.5, 0.75]
    traces["force_n"] = [120, 120, 120, 80]
    engine = {
        "method_spec": {"resolved_start_s": 0, "quiet_window_s": [0, 0.25]},
        "flights": [
            {"takeoff_s": 0.5, "landing_s": 0.75, "takeoff_velocity_m_s": 1},
            {"takeoff_s": 0.55, "landing_s": 0.74, "takeoff_velocity_m_s": 1.3},
        ],
        "diagnostic_only": True,
        "verdict": "reject",
        "qc": [{"status": "fail"}],
        "body_weight_n": 100,
        "mass_kg": 10,
        "onset": {"time_s": 0.25},
        "metrics": {"net_impulse_n_s": 10},
        "uncertainty": {"event_interval_resolution_s": 0.001},
    }
    model = {
        "mass_kg": 10,
        "gravity_m_s2": 10,
        "physics_times_s": [0, 0.25, 0.55, 0.75],
        "vertical_momentum_kg_m_s": [0, 5, 10, 0],
    }
    events = {
        key: {"time_s": t}
        for key, t in (
            ("takeoff", 0.55),
            ("touchdown", 0.75),
            ("kinetic_onset", 0.01),
        )
    }
    report = COMPARATOR["compare_case"](
        "test",
        "cmj",
        engine,
        traces,
        model,
        events,
        {"reference_resolution_s": 0.001, "synchronization_truth_check": {}},
    )
    conditions = report["comparisons"]
    assert report["diagnostic_only"]
    assert report["withheld"]["all_trial_metrics"]
    assert conditions["start_to_takeoff_net_impulse_n_s"]["observed_minus_reference"] == 0
    assert conditions["takeoff_s"]["observed"] == 0.5  # No truth-guided selection of flight 2.
    assert conditions["takeoff_s"]["within_diagnostic_tolerance"] is False
    assert conditions["whole_record_net_impulse_n_s"]["observed_minus_reference"] == 10
    assert conditions["onset_s_different_definitions"]["diagnostic_tolerance"] is None


def test_comparator_requires_both_run_receipts_before_opening_corpus(tmp_path):
    processing = tmp_path / "processing"
    processing.mkdir()
    (processing / "isolation.txt").write_text("SOURCE-only mount; test")
    with pytest.raises(FileNotFoundError, match=r"engine.json"):
        COMPARATOR["sealed_run"](processing, tmp_path / "protected", tmp_path / "missing", None)
