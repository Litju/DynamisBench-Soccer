"""DEVELOPMENT_ONLY_NOT_QUALIFIED reproducible offline demonstration."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from dynamisbench_soccer.m4.contracts import STAMP
from dynamisbench_soccer.m4.demo_fixtures import episodes
from dynamisbench_soccer.m4.demo_keys import demo_key
from dynamisbench_soccer.m4.key_store import digest, load_key
from dynamisbench_soccer.m4.runner import MAX_SOURCE, execute, trial

HEADER = """
import json, math, random, sys
episode = json.load(sys.stdin)
task = episode["task"]
x = task["inputs"]
answer = dict(status="DEVELOPMENT_ONLY_NOT_QUALIFIED", version="m4.answer.v1",
              decision="compute", reason="", method="", metrics={},
              estimand=task["estimand"], unit=task["unit"])
"""

ORACLE = """
if task["task_id"] == "fractional_cod":
    t0, t1 = x["time_s"][:2]
    v0, v1 = x["speed_m_s"][:2]
    a = (v1 - v0) / (t1 - t0)
    start = x["trigger_s"]
    start_speed = v0 + a * (start - t0)
    d = x["path_length_m"]
    elapsed = 2*d / (start_speed + math.sqrt(start_speed**2 + 2*a*d))
    partial = (start_speed + v1) * (t1-start) / 2
    answer.update(method="linear_speed_integral", reason="declared_fractional_start",
                  metrics=dict(elapsed_s=elapsed, first_interval_distance_m=partial))
elif task["task_id"] == "ballistic_height" and x["equal_com_height"]:
    speed = x["gravity_m_s2"] * x["flight_s"] / 2
    height = speed**2 / (2 * x["gravity_m_s2"])
    answer.update(method="velocity_energy", reason="equal_com_height",
                  metrics=dict(height_m=height, takeoff_speed_m_s=speed))
elif task["task_id"] == "gate_course" and x["positions_m"][0] != 0:
    answer.update(decision="refuse", method="course_geometry", reason="missing_course_origin")
elif task["task_id"] == "drop_jump" and x["independent_touchdown_velocity_m_s"] is None:
    answer.update(decision="refuse", method="dj_eligibility",
                  reason="missing_independent_touchdown")
"""

CONTROLS = {
    "oracle": ORACLE,
    "naive": """
answer.update(method="copy_or_zero", metrics={"elapsed_s": x.get("vendor_total_s", 0)})
""",
    "random": """
rng = random.Random(episode["seed"])
answer.update(method="guess", metrics={"height_m": rng.uniform(0, 5)})
""",
    "null": "answer = None\n",
}


def run(candidate: bytes | None = None) -> dict[str, Any]:
    planned = episodes()
    if execute(planned[0], b"print('sandbox ready')") != ("graded", b"sandbox ready\n"):
        raise RuntimeError(
            "M4 sandbox startup failed. Check bubblewrap and user namespace support."
        )
    sources = {
        name: (HEADER + body + "\nprint(json.dumps(answer))\n").encode()
        for name, body in CONTROLS.items()
    }
    if candidate is not None:
        sources = {"candidate": candidate}
    rows = []
    with tempfile.TemporaryDirectory(prefix="dbs-m4-grader-") as private:
        for episode in planned:
            raw = demo_key(episode).model_dump_json().encode()
            path = Path(private) / f"{episode.episode_id}.json"
            path.write_bytes(raw)
            key = load_key(path, digest(raw), episode)
            rows.append(
                {
                    "status": STAMP,
                    "task_id": episode.task.task_id,
                    "evidence": episode.task.evidence,
                    "episode_sha256": key.episode_sha256,
                    "scores": {
                        name: trial(episode, source, key).model_dump(mode="json")
                        for name, source in sources.items()
                    },
                }
            )
    return {
        "status": STAMP,
        "version": "m4.demo.v1",
        "candidate_sha256": {name: digest(source) for name, source in sources.items()},
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--candidate", type=Path, help="Python source to run as the candidate")
    mode.add_argument("--episodes", action="store_true", help="print the public episodes only")
    args = parser.parse_args()
    if args.episodes:
        value = {"status": STAMP, "episodes": [e.model_dump(mode="json") for e in episodes()]}
    else:
        candidate = None
        if args.candidate is not None:
            with args.candidate.open("rb") as stream:
                candidate = stream.read(MAX_SOURCE + 1)
            if len(candidate) > MAX_SOURCE:
                parser.error("candidate source exceeds size limit")
        value = run(candidate)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
