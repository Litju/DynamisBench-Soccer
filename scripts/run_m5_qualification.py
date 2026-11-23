"""Exercise M5 controls with a separately pinned, owner-private M4 catalog."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from seal_m5_references import _custody_family, _read_private

from dynamisbench_soccer.m4.admission import Catalog, evaluate
from dynamisbench_soccer.m4.grader import score_checkpoint
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.measurement import ENGINE
from dynamisbench_soccer.m4.runner import MAX_OUTPUT, MAX_SOURCE, WALL_SECONDS, execute_payload
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.m4.trials import STUBS, TrialContext, TrialReport, run_episode

ROOT = Path(__file__).resolve().parents[1]
SEED = 704


def _receipt(report: TrialReport) -> dict:
    receipt = asdict(report)
    receipt["admission"] = report.admission.model_dump(mode="json")
    for item in receipt["attempts"]:
        item["execution"]["stdout"] = item["execution"]["stdout"].decode(errors="replace")
        item["grade"] = item["grade"].model_dump(mode="json")
    return receipt


def _isolation(task, inputs, private):
    marker = secrets.token_hex(32)
    with tempfile.TemporaryFile(dir=private) as handle:
        handle.write(marker.encode())
        handle.flush()
        handle.seek(0)
        descriptor = handle.fileno()
        os.set_inheritable(descriptor, True)
        name = "DBS_M5_PRIVATE_CANARY"
        os.environ[name] = marker
        try:
            payload = task.candidate_payload(task.checkpoints[0].id, inputs, SEED)
            # Isolation checks observe namespace boundaries without retrieving private data.
            probe = f"""
import errno, os
from pathlib import Path
assert not Path({str(private)!r}).exists()
assert {name!r} not in os.environ
assert not Path('/proc/self').exists()
assert not Path('/proc/{os.getpid()}').exists()
assert os.getpid() != {os.getpid()}
try:
    os.fstat({descriptor})
except OSError as exc:
    assert exc.errno == errno.EBADF
else:
    raise AssertionError('host descriptor visible')
assert set(os.listdir('/')) == {{'candidate.py', 'usr', 'lib', 'lib64'}}
print('isolated')
"""
            result = execute_payload(payload, probe.encode())
        finally:
            del os.environ[name]
            os.set_inheritable(descriptor, False)
    return result


def _controls(score, task, answers: dict, inputs: dict):
    tolerance = task.methods[0].absolute_tolerance
    compute_id, positive = next(
        (key, row) for key, row in answers.items() if row["decision"] == "compute"
    )
    refuse_id, refusal = next(
        (key, row) for key, row in answers.items() if row["reason"] == "unsupported_claim"
    )
    quantity = positive["quantity"]
    reversed_claims = list(reversed(positive["claims"]))
    scenarios = [
        ("positive", compute_id, positive, ()),
        ("claim_order", compute_id, positive | {"claims": reversed_claims}, ()),
        (
            "inside_tolerance",
            compute_id,
            positive | {"quantity": quantity | {"value": quantity["value"] + 0.5 * tolerance}},
            (),
        ),
        (
            "numeric_error",
            compute_id,
            positive | {"quantity": quantity | {"value": quantity["value"] + 2 * tolerance}},
            ("value",),
        ),
        (
            "wrong_units",
            compute_id,
            positive | {"quantity": quantity | {"unit": "km/h", "dimension": "other"}},
            ("unit_or_dimension",),
        ),
        (
            "wrong_uncertainty",
            compute_id,
            positive | {"quantity": quantity | {"uncertainty": 0}},
            ("uncertainty",),
        ),
        (
            "wrong_estimand",
            compute_id,
            positive | {"quantity": quantity | {"estimand": "instantaneous_COM_metric"}},
            ("estimand",),
        ),
        ("wrong_method", compute_id, positive | {"method": "shortcut"}, ("unsupported_method",)),
        (
            "dropped_evidence",
            compute_id,
            positive | {"operands": positive["operands"][:-1]},
            ("derivation",),
        ),
        (
            "wrong_citation",
            compute_id,
            positive | {"citations": ["unrelated"]},
            ("source_citations",),
        ),
        (
            "invented_equivalence",
            compute_id,
            positive | {"claims": positive["claims"] + ["instruments_interchangeable"]},
            ("interpretation_boundary",),
        ),
        (
            "contradictory_free_text",
            compute_id,
            positive | {"explanation": "Interchangeability is established."},
            ("invalid_submission",),
        ),
        ("correct_refusal", refuse_id, refusal, ()),
        ("unnecessary_abstention", compute_id, refusal, ("eligibility_or_refusal",)),
        (
            "wrong_refusal_reason",
            refuse_id,
            refusal
            | {
                "reason": "not_eligible"
                if refusal["reason"] != "not_eligible"
                else "unsupported_claim"
            },
            ("eligibility_or_refusal",),
        ),
    ]
    for key, row in answers.items():
        if row["reason"] not in {"not_eligible", "insufficient_evidence"}:
            continue
        scenarios.extend(
            [
                (f"{key}_correct_qc", key, row, ()),
                (f"{key}_ignored_qc", key, positive, ("eligibility_or_refusal",)),
                (
                    f"{key}_wrong_qc_finding",
                    key,
                    row | {"claims": ["irrelevant_fault"]},
                    ("interpretation_boundary",),
                ),
            ]
        )

    method = task.methods[0].id
    if method == "beam-split":
        key = task.checkpoints[1].id
        candidate = answers[key]
        wrong = [
            operand
            | {
                "value": next(
                    item["value"] for item in candidate["operands"] if item["field"] == "split_s"
                )
            }
            if operand["field"] == "cumulative_s"
            else operand
            for operand in candidate["operands"]
        ]
        scenarios.append(
            ("omit_fixed_start_offset", key, candidate | {"operands": wrong}, ("derivation",))
        )
    elif method == "switch-flight-height":
        key = task.checkpoints[1].id
        candidate = answers[key]
        claims = ["active_high" if item == "active_low" else item for item in candidate["claims"]]
        scenarios.append(
            (
                "wrong_contact_polarity",
                key,
                candidate | {"claims": claims},
                ("interpretation_boundary",),
            )
        )
    elif method == "enu-positional-speed":
        (artifact,) = positive["selected_trials"]
        data = json.loads(inputs[artifact])
        doppler = data["channels"]["doppler_speed"][data["row"]]
        scenarios.append(
            (
                "substitute_native_doppler",
                compute_id,
                positive | {"quantity": quantity | {"value": doppler}},
                ("value",),
            )
        )
        key = task.checkpoints[1].id
        candidate = answers[key]
        (artifact,) = candidate["selected_trials"]
        data = json.loads(inputs[artifact])
        drift = data["meta"]["acquisition"]["alignments"][0]["drift_ppm"]
        target = candidate["quantity"]
        observed_clock = target | {"value": target["value"] * (1 + drift * 1e-6)}
        scenarios.append(
            ("ignore_affine_clock", key, candidate | {"quantity": observed_clock}, ("value",))
        )
    elif method == "paired-speed-bias":
        biased_id = task.checkpoints[1].id
        biased = answers[biased_id]
        signed = biased["quantity"]
        spread_id = task.checkpoints[2].id
        spread = answers[spread_id]
        scenarios.extend(
            [
                (
                    "reversed_bias_sign",
                    biased_id,
                    biased | {"quantity": signed | {"value": -signed["value"]}},
                    ("value",),
                ),
                (
                    "omitted_common_offset_uncertainty",
                    biased_id,
                    biased | {"quantity": signed | {"uncertainty": None}},
                    ("uncertainty",),
                ),
                (
                    "shrink_common_offset_by_pair_count",
                    biased_id,
                    biased
                    | {
                        "quantity": signed
                        | {
                            "uncertainty": signed["uncertainty"]
                            / (len(biased["selected_trials"]) ** 0.5)
                        }
                    },
                    ("uncertainty",),
                ),
                (
                    "bias_only_tolerance",
                    spread_id,
                    spread
                    | {
                        "claims": [
                            "limits_within_tolerance"
                            if claim == "limits_exceed_tolerance"
                            else claim
                            for claim in spread["claims"]
                        ]
                    },
                    ("interpretation_boundary",),
                ),
                (
                    "wrong_coordinate_frame",
                    biased_id,
                    biased | {"claims": biased["claims"] + ["sensor_frame_speed"]},
                    ("interpretation_boundary",),
                ),
                (
                    "wrong_pair_identity",
                    biased_id,
                    biased
                    | {"selected_trials": [*biased["selected_trials"][:-1], "unmatched-run"]},
                    ("trial_selection",),
                ),
                (
                    "wrong_acquisition_source",
                    biased_id,
                    biased
                    | {
                        "operands": [
                            biased["operands"][0] | {"artifact": "unmatched-source"},
                            *biased["operands"][1:],
                        ]
                    },
                    ("derivation",),
                ),
                (
                    "copied_radial_value",
                    biased_id,
                    biased | {"quantity": signed | {"value": biased["operands"][0]["value"] * 0.6}},
                    ("value",),
                ),
                (
                    "te_or_change_inferred_from_paired_devices",
                    biased_id,
                    biased
                    | {
                        "claims": biased["claims"]
                        + ["test_retest_typical_error_known", "individual_change_is_real"]
                    },
                    ("interpretation_boundary",),
                ),
                (
                    "compute_after_definition_mismatch",
                    refuse_id,
                    positive,
                    ("eligibility_or_refusal",),
                ),
            ]
        )
    else:
        scenarios.append(
            (
                "drop_filter_contrast_sign",
                compute_id,
                positive | {"quantity": quantity | {"value": abs(quantity["value"])}},
                ("value",),
            )
        )

    results = {}
    for name, checkpoint, candidate, findings in scenarios:
        raw = json.dumps(candidate).encode()
        grade = score(checkpoint, raw)
        valid = grade.outcome == ("FAIL" if findings else "PASS") and set(findings) <= set(
            grade.findings
        )
        results[name] = {
            "expected": "FAIL" if findings else "PASS",
            "actual": grade.outcome,
            "findings": grade.findings,
            "passed": valid,
            "repeatable": grade == score(checkpoint, raw),
        }
    return results


def _stable(report: TrialReport) -> tuple:
    return (
        report.admission,
        report.terminal_reason,
        tuple(
            (
                a.checkpoint,
                a.candidate_payload_sha256,
                a.output_sha256,
                a.grade,
                a.execution.outcome,
                a.execution.workspace_removed,
            )
            for a in report.attempts
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("custody", type=Path)
    parser.add_argument("catalog_sha256")
    parser.add_argument("--task", help="limit the campaign to this reviewed measurement task ID")
    args = parser.parse_args()
    private = args.custody.resolve(strict=True)
    if (
        private.is_relative_to(ROOT)
        or ROOT.is_relative_to(private)
        or private.stat().st_uid != os.getuid()
        or private.stat().st_mode & 0o077
    ):
        parser.error("custody must be a disjoint owner-private directory")
    if not re.fullmatch(r"[0-9a-f]{64}", args.catalog_sha256):
        parser.error("catalog SHA-256 must be lowercase hexadecimal")
    catalog = _read_private(private, f"catalog-{args.catalog_sha256}.json")
    context = TrialContext(ROOT, catalog, args.catalog_sha256, private)
    if digest(catalog) != context.trusted_catalog_sha256:
        parser.error("untrusted catalog digest")
    reviewed = Catalog.model_validate_json(catalog)
    registry = load_registry(ROOT / "benchmark/tasks")
    summary = {}
    passed = True
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=private, prefix="trials-", suffix=".jsonl", delete=False
    ) as receipt:
        for task in registry.values():
            if (
                task.engine != ENGINE
                or task.reference_profile not in reviewed.slices
                or (args.task and task.task.id != args.task)
            ):
                continue
            key = f"{task.task.id}@{task.task.revision}"
            inputs = {
                artifact.name: (ROOT / artifact.path).read_bytes() for artifact in task.artifacts
            }
            family = _custody_family(task)
            profile = reviewed.slices[task.reference_profile]
            oracle_source = _read_private(private, f"{family}-oracle.py")
            if digest(oracle_source) != profile.private_inputs_sha256.get("oracle"):
                parser.error("oracle differs from the reviewed custody digest")
            baselines = STUBS | {
                "naive": (ROOT / "scripts/m5_controls/naive.py").read_bytes(),
                "oracle": oracle_source,
            }
            rows = {
                name: run_episode(context, task, name, source, SEED)
                for name, source in baselines.items()
            }
            repeats = {
                name: run_episode(context, task, name, source, SEED)
                for name, source in baselines.items()
            }
            for label, trial_set in (("initial", rows), ("repeat", repeats)):
                for report in trial_set.values():
                    receipt.write(
                        json.dumps(
                            {
                                "head": head,
                                "revision": task.task.revision,
                                "artifacts": {
                                    artifact.name: artifact.sha256 for artifact in task.artifacts
                                },
                                "run": label,
                                "trial": _receipt(report),
                            }
                        )
                        + "\n"
                    )
            oracle = rows["oracle"]
            task_report = {
                "admission": oracle.admission.status,
                "reasons": oracle.admission.reasons,
                "task_sha256": oracle.admission.task_sha256,
                "source_digests": {artifact.name: artifact.sha256 for artifact in task.artifacts},
                "receipt_digests": oracle.admission.evidence,
                "score": {
                    name: dict(Counter(a.grade.outcome for a in row.attempts))
                    for name, row in rows.items()
                },
                "execution_errors": {
                    name: sum(a.execution.outcome != "graded" for a in row.attempts)
                    for name, row in rows.items()
                },
                "cleanup_errors": sum(
                    not a.execution.workspace_removed for row in rows.values() for a in row.attempts
                ),
                "replay_differences": sum(
                    _stable(row) != _stable(repeats[name]) for name, row in rows.items()
                ),
                "environment": oracle.environment,
                "oracle_sha256": oracle.candidate_sha256,
            }
            summary[key] = task_report
            if oracle.admission.status != "ENGINEERING_ADMITTED" or not oracle.attempts:
                passed = False
                continue
            valid = task_report["cleanup_errors"] == task_report["replay_differences"] == 0 and all(
                row.terminal_reason is None
                and len(row.attempts) == len(task.checkpoints)
                and all(
                    a.execution.outcome == "graded"
                    and a.grade.outcome == ("PASS" if name == "oracle" else "FAIL")
                    for a in row.attempts
                )
                for name, row in rows.items()
            )
            if not valid:
                passed = False
                continue
            answers = {a.checkpoint: json.loads(a.execution.stdout) for a in oracle.attempts}

            def score(checkpoint, raw, task=task, inputs=inputs):
                return score_checkpoint(
                    ROOT, task, catalog, args.catalog_sha256, private, checkpoint, inputs, raw
                )

            checks = _controls(score, task, answers, inputs)
            isolation = _isolation(task, inputs, private)
            leaked = not (
                isolation.outcome == "graded"
                and isolation.stdout == b"isolated\n"
                and isolation.workspace_removed
            )
            wrong_artifact = task.artifacts[0].model_copy(update={"sha256": "0" * 64})
            stale_input = task.model_copy(
                update={"artifacts": (wrong_artifact, *task.artifacts[1:])}
            )
            stale_role = next(iter(profile.evidence))
            stale_receipt = profile.evidence[stale_role].model_copy(update={"sha256": "0" * 64})
            stale_review = profile.model_copy(
                update={"evidence": profile.evidence | {stale_role: stale_receipt}}
            )
            stale_catalog = (
                Catalog(slices={task.reference_profile: stale_review}).model_dump_json().encode()
            )
            admission_controls = {
                "invalid_pin": evaluate(ROOT, task, catalog, "0" * 64).status,
                "stale_artifact": evaluate(ROOT, stale_input, catalog, args.catalog_sha256).status,
                "stale_receipt": evaluate(ROOT, task, stale_catalog, digest(stale_catalog)).status,
            }
            rejected = set(admission_controls.values()) == {"REFUSED"}
            task_report |= {
                "controls": checks,
                "false_pass": sum(
                    r["actual"] == "PASS" and r["expected"] == "FAIL" for r in checks.values()
                ),
                "false_fail": sum(
                    r["actual"] != "PASS" and r["expected"] == "PASS" for r in checks.values()
                ),
                "isolation": "FAIL" if leaked else "PASS",
                "admission_controls": admission_controls,
            }
            passed &= (
                not leaked
                and rejected
                and all(r["passed"] and r["repeatable"] for r in checks.values())
            )
    print(
        json.dumps(
            {
                "head": head,
                "catalog_sha256": args.catalog_sha256,
                "seed": SEED,
                "runner_limits": {
                    "wall_s": WALL_SECONDS,
                    "max_output_bytes": MAX_OUTPUT,
                    "max_source_bytes": MAX_SOURCE,
                },
                "tasks": summary,
                "passed": passed and bool(summary),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed or not summary:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
