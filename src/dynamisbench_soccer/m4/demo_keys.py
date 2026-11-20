"""DEVELOPMENT_ONLY_NOT_QUALIFIED independent demo oracle keys for the host grader."""

from __future__ import annotations

from dynamisbench_soccer.m4.contracts import Answer, Episode
from dynamisbench_soccer.m4.key_store import PrivateKey, digest


def demo_key(episode: Episode) -> PrivateKey:
    # Closed-form expectations are independent of candidate code and device engines.
    decision, method, reason, metrics = {
        "fractional_cod": (
            "compute",
            "linear_speed_integral",
            "declared_fractional_start",
            {"elapsed_s": 1.5, "first_interval_distance_m": 0.1875},
        ),
        "ballistic_height": (
            "compute",
            "flight_time",
            "equal_com_height",
            {"height_m": 0.44129925, "takeoff_speed_m_s": 2.941995},
        ),
        "partial_gates": ("refuse", "course_geometry", "missing_course_origin", {}),
        "missing_drop_evidence": ("refuse", "dj_eligibility", "missing_independent_touchdown", {}),
    }[episode.task.task_id]
    return PrivateKey(
        episode_sha256=digest(episode.model_dump_json().encode()),
        answer=Answer.model_validate(
            {
                "decision": decision,
                "method": method,
                "reason": reason,
                "metrics": metrics,
                "estimand": episode.task.estimand,
                "unit": episode.task.unit,
            }
        ),
        alternative_methods=("velocity_energy",) if episode.task.convention == "defended" else (),
        atol=1e-8,
    )
