"""DEVELOPMENT_ONLY_NOT_QUALIFIED independent demo oracle keys for the host grader."""

from __future__ import annotations

from dynamisbench_soccer.m4.contracts import Answer, Episode
from dynamisbench_soccer.m4.key_store import PrivateKey, digest

# Reviewed fixture bindings in (public episode, private key) order.
KEY_DIGESTS = {
    "fractional_cod": (
        "5bf969a346688c41e7ec3c4800546372f7570b8a046d1ceeb9e5d27ed457d1ed",
        "6ce748e531ee897ba2cacc9ea73255b7faca91b03fba8a01ded5cff417803293",
    ),
    "ballistic_height": (
        "80abcd83e8a61d93809e03f4a04af9880dfceaacf70bb20450bd1ec4bdbd4141",
        "3eddbc6f6d1e61c62c1865a4c238e2bfc22946db93864f7dc699f2a9a7b03a1a",
    ),
    "gate_course": (
        "dc370c39e32249cd5d41176e40978fb2f3ec92d4d652d27e9b9e9c35de890156",
        "085a6a053d97ae7c972e14af891939ed5028e351a1952bd91b7855b57142e619",
    ),
    "drop_jump": (
        "297ad3934ef643721e558d7fa3e03d63a268014a2f3ddf294428daf46d6eb5c6",
        "489fa653e425bca5e399b0869dd73e2fe87f6d62ccc20d6b14b8237230f28998",
    ),
}


def demo_key(episode: Episode) -> PrivateKey:
    public_sha, _ = KEY_DIGESTS[episode.task.task_id]
    if digest(episode.model_dump_json().encode()) != public_sha:
        raise ValueError("public episode does not match the reviewed demo fixture")
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
        "gate_course": ("refuse", "course_geometry", "missing_course_origin", {}),
        "drop_jump": ("refuse", "dj_eligibility", "missing_independent_touchdown", {}),
    }[episode.task.task_id]
    return PrivateKey(
        episode_sha256=public_sha,
        answer=Answer.model_validate(
            {
                "decision": decision,
                "method": method,
                "reason": reason,
                "metrics": metrics,
                "estimand": episode.task.estimand,
                "unit": episode.task.unit,
                "formula": "g*t**2/8" if episode.task.convention == "defended" else None,
            }
        ),
        alternative_formulas=(
            {"velocity_energy": "v**2/(2*g)"} if episode.task.convention == "defended" else {}
        ),
        atol=episode.task.absolute_tolerance,
    )
