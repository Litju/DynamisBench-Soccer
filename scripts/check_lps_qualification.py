"""Run the admitted LPS qualification case using externally sealed answer custody."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from check_m4_controls import Controls, evaluate_controls

from dynamisbench_soccer.m4.admission import Catalog, evaluate
from dynamisbench_soccer.m4.grader import score_checkpoint
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.runner import execute_payload
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.m4.trials import STUBS, TrialContext, run_episode, run_oracle_control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("custody", type=Path)
    parser.add_argument("catalog_sha256")
    parser.add_argument("withheld_controls", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    private = args.custody.resolve(strict=True)
    if private.is_relative_to(root) or private.stat().st_mode & 0o077:
        parser.error("custody must be an external private directory")
    controls_path = args.withheld_controls.resolve(strict=True)
    if controls_path.is_relative_to(root) or controls_path.stat().st_mode & 0o077:
        parser.error("diagnostic controls must be external and private")
    legacy = evaluate_controls(Controls.model_validate_json(controls_path.read_bytes()))
    catalog = (private / "catalog.json").read_bytes()
    context = TrialContext(root, catalog, args.catalog_sha256, private)
    task = load_registry(root / "benchmark/tasks")["native-lps-speed", 1]
    inputs = {a.name: (root / a.path).read_bytes() for a in task.artifacts}
    source = (root / "scripts/m4_controls/lps_oracle.py").read_bytes()
    oracle = run_oracle_control(context, task, source, seed=17)
    replay = run_oracle_control(context, task, source, seed=17)
    baselines = {"oracle": oracle}
    for name, stub in STUBS.items():
        baselines[name] = run_episode(context, task, name, stub, seed=17)
    with (private / "trial-receipts.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {name: asdict(report) for name, report in baselines.items()},
            stream,
            default=lambda raw: raw.decode("utf-8", errors="replace"),
        )

    clean, invalid = (json.loads(attempt.execution.stdout) for attempt in oracle.attempts)
    track = json.loads(inputs["lps-observation"])
    row = track["row"]
    quantity = clean["quantity"]

    def check(candidate: dict, checkpoint: str) -> str:
        return score_checkpoint(
            root,
            task,
            catalog,
            args.catalog_sha256,
            private,
            checkpoint,
            inputs,
            json.dumps(candidate).encode(),
        ).outcome

    higher_dimensional = (
        quantity["value"] ** 2
        + (
            (track["channels"]["position_z"][row + 1] - track["channels"]["position_z"][row - 1])
            / (track["channels"]["time"][row + 1] - track["channels"]["time"][row - 1])
        )
        ** 2
    ) ** 0.5
    cases = {
        "positive": [
            (check(clean, "clean"), True),
            (
                check(
                    clean | {"explanation": "Centered displacement norm divided by measured span."},
                    "clean",
                ),
                True,
            ),
        ],
        "negative": [
            (check(clean | {"quantity": quantity | {"value": higher_dimensional}}, "clean"), False),
            (check(clean | {"quantity": quantity | {"unit": "km/h"}}, "clean"), False),
            (check(clean | {"operands": clean["operands"][:-1]}, "clean"), False),
        ],
        "adversarial": [
            (check(clean | {"quantity": quantity | {"estimand": "doppler_speed"}}, "clean"), False),
            (check(clean | {"method": "one-sided"}, "clean"), False),
        ],
        "refusal": [
            (check(invalid, "invalid"), True),
            (check(clean, "invalid"), False),
            (check(invalid, "clean"), False),
        ],
    }
    counts = {}
    for category, rows in cases.items():
        counts[category] = dict(
            expected_accept=sum(expected for _, expected in rows),
            expected_reject=sum(not expected for _, expected in rows),
            false_accept=sum(outcome == "PASS" and not expected for outcome, expected in rows),
            false_reject=sum(outcome != "PASS" and expected for outcome, expected in rows),
            excluded=sum(outcome == "EXCLUDED_FROM_SCORED_BENCHMARK" for outcome, _ in rows),
        )

    review = Catalog.model_validate_json(catalog).slices[task.reference_profile]
    bad_receipt = review.model_copy(
        update={
            "evidence": {
                **review.evidence,
                "H": review.evidence["H"].model_copy(update={"sha256": "0" * 64}),
            }
        }
    )
    stale_catalog = Catalog(slices={task.reference_profile: bad_receipt}).model_dump_json().encode()
    bad_source = task.model_copy(
        update={"task": task.task.model_copy(update={"source": "gnss_lps.gnss"})}
    )
    changed_artifact = task.artifacts[0].model_copy(update={"sha256": "0" * 64})
    tampered = task.model_copy(update={"artifacts": (changed_artifact, *task.artifacts[1:])})
    rejections = [
        evaluate(root, bad_source, catalog, args.catalog_sha256),
        evaluate(root, tampered, catalog, args.catalog_sha256),
        evaluate(root, task, stale_catalog, digest(stale_catalog)),
        evaluate(root, task, catalog, "0" * 64),
    ]
    false_admits = sum(x.status != "REFUSED" for x in rejections)
    private_path = private / f"{review.answer.id}.json"
    probe = f"from pathlib import Path\nassert not Path({str(private_path)!r}).exists()\nprint('isolated')".encode()
    projection = task.candidate_payload("clean", inputs, 17)
    isolation = execute_payload(projection, probe)
    keys_hidden = isolation.stdout == b"isolated\n" and isolation.workspace_removed
    differences = sum(
        a.output_sha256 != b.output_sha256
        for a, b in zip(oracle.attempts, replay.attempts, strict=True)
    )
    summary = {
        "scope": task.task.id,
        "admission": oracle.admission.status,
        "receipts": oracle.admission.evidence,
        "admission_positive": 1,
        "admission_false_reject": 0,
        "admission_negative": len(rejections),
        "admission_false_accept": false_admits,
        "baselines": {
            name: dict(Counter(attempt.grade.outcome for attempt in report.attempts))
            for name, report in baselines.items()
        },
        "key_isolation": keys_hidden,
        "replay_differences": differences,
        "native_controls": counts,
        "diagnostic_controls": legacy,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    passed = (
        legacy["diagnostic_thresholds_met"]
        and not false_admits
        and keys_hidden
        and not differences
        and all(
            len(cases[key]) >= 2
            and not any(result[item] for item in ("false_accept", "false_reject", "excluded"))
            for key, result in counts.items()
        )
        and all(
            report.terminal_reason is None
            and len(report.attempts) == len(task.checkpoints)
            and all(
                attempt.execution.workspace_removed and attempt.grade.outcome == "FAIL"
                for attempt in report.attempts
            )
            for name, report in baselines.items()
            if name != "oracle"
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
