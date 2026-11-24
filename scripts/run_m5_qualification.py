"""Exercise M5 controls with a separately pinned, owner-private M4 catalog."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from seal_m5_references import _custody_family, _read_private, _supported

from dynamisbench_soccer.m4.admission import Catalog, evaluate
from dynamisbench_soccer.m4.grader import score_checkpoint
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.measurement import (
    CELL_OPERATION,
    IMU_POSITION,
    IMU_SIGNATURE,
    LPT_OPERATION,
)
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
        ((key, row) for key, row in answers.items() if row["reason"] == "unsupported_claim"),
        next((key, row) for key, row in answers.items() if row["decision"] == "refuse"),
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
            positive
            | {
                (
                    "unsupported_field" if task.grader == "formal.v1" else "explanation"
                ): "Interchangeability is established."
            },
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
        positive_for_method = (
            next(
                value
                for value in answers.values()
                if value["decision"] == "compute" and value["method"] == row["method"]
            )
            if task.methods[0].id == IMU_POSITION
            else positive
        )
        scenarios.extend(
            [
                (f"{key}_correct_qc", key, row, ()),
                (f"{key}_ignored_qc", key, positive_for_method, ("eligibility_or_refusal",)),
                (
                    f"{key}_wrong_qc_finding",
                    key,
                    row | {"claims": ["irrelevant_fault"]},
                    ("interpretation_boundary",),
                ),
            ]
        )

    method = task.methods[0].id
    if method == LPT_OPERATION:
        data = json.loads(inputs[positive["selected_trials"][0]])
        angle = data["meta"]["acquisition"]["sources"][0]["geometry"]["cable_angle_rad"]
        cable_speed = next(
            o["value"] for o in positive["operands"] if o["field"] == "cable_speed_m_s"
        )
        clock_scale = data["clock_scale_bound_ppm"] * 1e-6
        time, length = data["channels"]["time"], data["channels"]["cable_length"]
        row = data["row"]
        forward = (length[row + 1] - length[row]) / ((time[row + 1] - time[row]) * math.cos(angle))
        for name, value in (
            ("wrong_projection_angle", cable_speed / math.cos(angle + data["angle_bound_rad"])),
            ("cable_speed_as_vertical", cable_speed),
            ("reversed_attachment_direction", -quantity["value"]),
            ("clock_corner_as_nominal", quantity["value"] / (1 + clock_scale)),
            ("forward_derivative_stencil", forward),
        ):
            scenarios.append(
                (name, compute_id, positive | {"quantity": quantity | {"value": value}}, ("value",))
            )
        for name, field, value in (
            ("length_as_relative_displacement", "vertical_displacement_m", length[row]),
            ("acceleration_without_projection", "vertical_acceleration_m_s2", 1.6),
            ("power_with_gravity_omitted", "vertical_mass_model_power_w", 20 * 2 * 0.52),
            ("power_with_wrong_mass", "vertical_mass_model_power_w", 122.78916 / 2),
        ):
            operands = [
                operand | {"value": value} if operand["field"] == field else operand
                for operand in positive["operands"]
            ]
            scenarios.append((name, compute_id, positive | {"operands": operands}, ("derivation",)))
        scenarios.extend(
            (
                (
                    "adjacent_operands_unordered",
                    compute_id,
                    positive | {"operands": positive["operands"][::-1]},
                    (),
                ),
                (
                    "missing_clock_uncertainty",
                    compute_id,
                    positive
                    | {
                        "quantity": quantity
                        | {
                            "uncertainty": cable_speed / math.cos(angle + data["angle_bound_rad"])
                            - quantity["value"]
                        }
                    },
                    ("uncertainty",),
                ),
                (
                    "attachment_as_com",
                    compute_id,
                    positive | {"quantity": quantity | {"estimand": "athlete_com_vertical_speed"}},
                    ("estimand",),
                ),
                (
                    "unsupported_com_power",
                    compute_id,
                    positive | {"claims": positive["claims"] + ["athlete_com_power_observed"]},
                    ("interpretation_boundary",),
                ),
                (
                    "unsupported_force_plate_power",
                    refuse_id,
                    positive | {"claims": positive["claims"] + ["force_platform_power_observed"]},
                    ("eligibility_or_refusal", "interpretation_boundary"),
                ),
            )
        )
    elif method == IMU_POSITION:
        signature_id, signature = next(
            (key, row)
            for key, row in answers.items()
            if row["method"] == IMU_SIGNATURE and row["decision"] == "compute"
        )
        gap = next(row for row in answers.values() if row["claims"] == ["native_clock_gap"])
        sig_quantity = signature["quantity"]
        artifact = positive["selected_trials"][0]
        native_case = json.loads(inputs[artifact])
        channels = native_case["channels"]
        spec = native_case["spec"]

        for name, key, answer, field, expected_wrong in (
            ("specific_force_as_acceleration", compute_id, positive, "free_az_m_s2", 11.00665),
            ("gravity_wrong_sign", compute_id, positive, "free_az_m_s2", 20.8133),
            ("wrong_frame", compute_id, positive, "free_ax_m_s2", -1),
            ("ignored_accel_bias", compute_id, positive, "free_ax_m_s2", 2.2),
            ("ignored_accel_scale", compute_id, positive, "free_ax_m_s2", 1.78),
            ("misalignment_not_inverted", compute_id, positive, "free_ay_m_s2", -1.4),
            ("misalignment_applied_forward", compute_id, positive, "free_ay_m_s2", -1.8),
            (
                "gyro_bias_uncorrected",
                compute_id,
                positive,
                "peak_corrected_gyro_rad_s",
                math.hypot(*spec["gyro_bias_rad_s"]),
            ),
            ("missed_velocity_integration", compute_id, positive, "terminal_vx_m_s", 0.3),
            ("first_low_sample_edge", signature_id, signature, "takeoff_edge_s", 0.1),
            ("missing_signature_bracket", signature_id, signature, "landing_edge_s", None),
            ("wrong_threshold", signature_id, signature, "threshold_m_s2", 4),
        ):
            operands = [
                operand | {"value": expected_wrong} if operand["field"] == field else operand
                for operand in answer["operands"]
                if operand["field"] != field or expected_wrong is not None
            ]
            scenarios.append((name, key, answer | {"operands": operands}, ("derivation",)))

        for name, key, answer, changes, findings in (
            (
                "position_operand_order",
                compute_id,
                positive,
                {"operands": positive["operands"][::-1]},
                (),
            ),
            (
                "position_as_com",
                compute_id,
                positive,
                {"quantity": quantity | {"estimand": "athlete_com_position"}},
                ("estimand",),
            ),
            (
                "wrong_source_trial",
                compute_id,
                positive,
                {"selected_trials": signature["selected_trials"]},
                ("trial_selection",),
            ),
            (
                "position_method_substitution",
                compute_id,
                positive,
                {"method": IMU_SIGNATURE},
                ("unsupported_method",),
            ),
            (
                "attitude_from_gravity_norm",
                "c02",
                positive,
                {"claims": ["attitude_from_gravity_norm"]},
                ("eligibility_or_refusal",),
            ),
            ("signature_positive", signature_id, signature, {}, ()),
            (
                "signature_inside_tolerance",
                signature_id,
                signature,
                {"quantity": sig_quantity | {"value": sig_quantity["value"] + 0.5 * tolerance}},
                (),
            ),
            (
                "signature_operand_order",
                signature_id,
                signature,
                {"operands": signature["operands"][::-1]},
                (),
            ),
            (
                "signature_wrong_duration",
                signature_id,
                signature,
                {"quantity": sig_quantity | {"value": sig_quantity["value"] - 0.05}},
                ("value",),
            ),
            (
                "signature_missing_second_edge_uncertainty",
                signature_id,
                signature,
                {"quantity": sig_quantity | {"uncertainty": 0.05}},
                ("uncertainty",),
            ),
            (
                "signature_wrong_units",
                signature_id,
                signature,
                {"quantity": sig_quantity | {"unit": "ms"}},
                ("unit_or_dimension",),
            ),
            (
                "signature_wrong_citation",
                signature_id,
                signature,
                {"citations": ["DEV-003"]},
                ("source_citations",),
            ),
            (
                "signature_as_physical_flight",
                signature_id,
                signature,
                {"quantity": sig_quantity | {"estimand": "physical_flight_time"}},
                ("estimand",),
            ),
            (
                "signature_force_platform_claim",
                signature_id,
                signature,
                {"claims": signature["claims"] + ["force_platform_events_observed"]},
                ("interpretation_boundary",),
            ),
            (
                "signature_method_substitution",
                signature_id,
                signature,
                {"method": IMU_POSITION},
                ("unsupported_method",),
            ),
            (
                "signature_unnecessary_abstention",
                signature_id,
                gap,
                {},
                ("eligibility_or_refusal",),
            ),
            (
                "position_without_initial_velocity",
                compute_id,
                positive,
                {"quantity": quantity | {"value": 1.66}},
                ("value",),
            ),
            (
                "native_sensor_axis_shortcut",
                compute_id,
                positive,
                {"quantity": quantity | {"value": channels["specific_force_x"][0]}},
                ("value",),
            ),
            ("calculate_com_on_invalid_case", "c06", positive, {}, ("eligibility_or_refusal",)),
        ):
            scenarios.append((name, key, answer | changes, findings))
    elif method == CELL_OPERATION:
        sensitivity_id = task.checkpoints[1].id
        sensitivity = answers[sensitivity_id]
        for name, checkpoint, answer, field, replacement in (
            ("wrong_calibration_gain", compute_id, positive, "gross_peak_n", 612),
            ("wrong_calibration_offset", compute_id, positive, "baseline_n", 132),
            ("uninverted_sensitivity", sensitivity_id, sensitivity, "net_peak_n", 0.0024),
            ("wrong_moment_arm", compute_id, positive, "lever_arm_m", 0.42),
            ("wrong_baseline", compute_id, positive, "net_peak_n", 812),
            ("wrong_onset", compute_id, positive, "onset_s", 0.121),
            ("wrong_rfd_window", compute_id, positive, "rfd_100ms_n_s", 8000 / 3),
            ("missing_impulse", compute_id, positive, "impulse_100ms_n_s", None),
        ):
            operands = [
                operand | {"value": replacement} if operand["field"] == field else operand
                for operand in answer["operands"]
                if operand["field"] != field or replacement is not None
            ]
            scenarios.append((name, checkpoint, answer | {"operands": operands}, ("derivation",)))
        scenarios.extend(
            [
                (
                    "operand_order",
                    compute_id,
                    positive | {"operands": positive["operands"][::-1]},
                    (),
                ),
                (
                    "force_as_torque",
                    compute_id,
                    positive | {"quantity": quantity | {"unit": "N", "dimension": "force"}},
                    ("unit_or_dimension",),
                ),
                (
                    "radius_without_product",
                    compute_id,
                    positive
                    | {"quantity": quantity | {"uncertainty": quantity["uncertainty"] - 0.02}},
                    ("uncertainty",),
                ),
                (
                    "wrong_trial",
                    compute_id,
                    positive | {"selected_trials": task.checkpoints[1].artifacts},
                    ("trial_selection",),
                ),
                (
                    "internal_muscle_force",
                    compute_id,
                    positive | {"claims": positive["claims"] + ["internal_muscle_force"]},
                    ("interpretation_boundary",),
                ),
            ]
        )
    elif method == "beam-split":
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
    elif method == "three-point":
        scenarios.append(
            (
                "equivalent_explanation",
                compute_id,
                positive
                | {"explanation": "Horizontal symmetric difference, excluding vertical motion."},
                (),
            )
        )
    elif method == "fixed-session-change":
        data = json.loads(inputs[positive["selected_trials"][0]])
        sizes = [len(session["trials_m_s"]) for session in data["baseline"][:3]]
        n_current = len(data["current"]["trials_m_s"])
        evidence = {o["field"]: o["value"] for o in positive["operands"]}
        pooled_se = evidence["typical_error"] * (1 / n_current + 1 / sum(sizes)) ** 0.5

        def altered_operand(field, value):
            return positive | {
                "operands": [
                    o | {"value": value} if o["field"] == field else o for o in positive["operands"]
                ]
            }

        scenarios.extend(
            [
                (
                    "pooled_trial_uncertainty",
                    compute_id,
                    altered_operand("change_se", pooled_se),
                    ("derivation",),
                ),
                (
                    "wrong_cohort_typical_error",
                    compute_id,
                    altered_operand("typical_error", evidence["typical_error"] * 2**0.5),
                    ("derivation",),
                ),
                (
                    "wrong_interval_limit",
                    compute_id,
                    altered_operand("interval_upper", evidence["interval_upper"] + 0.001),
                    ("derivation",),
                ),
                (
                    "inside_operand_tolerance",
                    compute_id,
                    altered_operand("change_se", evidence["change_se"] + tolerance / 2),
                    (),
                ),
                (
                    "wrong_swc_decision",
                    compute_id,
                    positive
                    | {
                        "claims": [
                            "point_below_swc" if claim == "point_reaches_swc" else claim
                            for claim in positive["claims"]
                        ]
                    },
                    ("interpretation_boundary",),
                ),
                (
                    "wrong_signed_change",
                    compute_id,
                    positive | {"quantity": quantity | {"value": -quantity["value"]}},
                    ("value",),
                ),
                (
                    "claims_empirical_reliability",
                    compute_id,
                    positive | {"claims": [*positive["claims"], "empirical_reliability"]},
                    ("interpretation_boundary",),
                ),
                (
                    "includes_te_estimation_uncertainty",
                    compute_id,
                    positive
                    | {
                        "claims": [
                            claim
                            for claim in positive["claims"]
                            if claim != "te_estimation_uncertainty_excluded"
                        ]
                    },
                    ("interpretation_boundary",),
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
                not _supported(task)
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
                "naive": STUBS["naive"]
                if family == "native"
                else (ROOT / "scripts/m5_controls/naive.py").read_bytes(),
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
                    and a.grade.outcome in {"PASS", "FAIL"}
                    and (name != "oracle" or a.grade.outcome == "PASS")
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
