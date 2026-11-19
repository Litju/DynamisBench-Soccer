"""Post-run comparison for the RES-718 FORCE-only pilot. Opens protected model references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamisbench_soccer.adapters.biosyngen_force import CASES, UPSTREAM
from dynamisbench_soccer.data.manifest import DatasetManifest, sha256_file
from dynamisbench_soccer.engines.force_platform_acquisition import input_digest
from dynamisbench_soccer.signals.io import read


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def discrepancy(observed, reference, tolerance=None):
    delta = observed - reference if observed is not None else None
    return {
        "observed": observed,
        "modeled_reference": reference,
        "observed_minus_reference": delta,
        "diagnostic_tolerance": tolerance,
        "within_diagnostic_tolerance": abs(delta) <= tolerance
        if delta is not None and tolerance is not None
        else None,
    }


def sealed_run(processing, corpus, verification, dataset):
    require(
        (processing / "isolation.txt").read_text().startswith("SOURCE-only mount;"),
        "missing isolated processing receipt",
    )
    # Complete and bind both production runs before opening the corpus.
    reports = {}
    for case, protocol in CASES.items():
        run = processing / case
        report = load(run / "engine.json")
        bundle = read(run / "force")
        require(
            bundle.meta.bundle_id == case
            and bundle.meta.test_id == protocol
            and report["engine"] == "force-platform.v1"
            and report["method_spec"]["input_digests"] == [input_digest(bundle)] * 2
            and report["trace_output"]["sha256"] == sha256_file(run / "traces.csv")
            and int((run / "exit-code").read_text()) == int(report["verdict"] == "reject"),
            "processor report or traces do not bind to the completed trial",
        )
        approved = {f.path: f.sha256 for f in dataset.files}
        require(
            bundle.meta.provenance.source_sha256 == approved[f"{case}/streams/force.csv"]
            and f"source_manifest_sha256={approved[f'{case}/manifest.json']}"
            in bundle.meta.provenance.transforms,
            "converted trial does not bind approved source",
        )
        traces = np.genfromtxt(run / "traces.csv", delimiter=",", names=True)
        for column, raw in (("fz_plate1_n", "fz_left"), ("fz_plate2_n", "fz_right")):
            require(np.array_equal(traces[column], bundle[raw]), "bilateral trace identity differs")
        require(
            np.array_equal(traces["raw_force_n"], bundle["fz_left"] + bundle["fz_right"]),
            "bilateral sum does not match raw trace",
        )
        reports[case] = (report, traces, sha256_file(run / "engine.json"))

    inventory = load(corpus / "inventory.json")
    actual = {f.relative_to(corpus).as_posix(): f for f in corpus.rglob("*") if f.is_file()}
    require(
        set(actual) == set(inventory) | {"inventory.json", "run.json"},
        "corpus inventory is incomplete or has extra files",
    )
    for name, digest in inventory.items():
        f = actual[name]
        require(
            not f.is_symlink()
            and f.resolve() == corpus.resolve() / name
            and sha256_file(f) == digest,
            f"corpus path or checksum differs: {name}",
        )
    run_receipt = load(corpus / "run.json")
    verified = load(verification)
    required_counts = (
        "requested_count",
        "source_integrity_pass_count",
        "model_reference_qualified_count",
        "bounded_synchronization_count",
        "full_case_qualified_count",
    )
    require(
        run_receipt["requested_count"] == run_receipt["generated_count"] == len(CASES)
        and run_receipt["refused_count"] == 0
        and all(verified[k] == len(CASES) for k in required_counts)
        and {c["case_id"]: c["protocol"] for c in run_receipt["cases"]} == CASES
        and {c["case_id"]: c["protocol"] for c in verified["cases"]} == CASES,
        "upstream generation or verification denominator differs",
    )
    return reports, run_receipt, verified


def compare_case(case, protocol, report, traces, model, events, verification):
    mass = model["mass_kg"]
    weight = mass * model["gravity_m_s2"]
    physics_t = np.asarray(model["physics_times_s"])
    momentum = np.asarray(model["vertical_momentum_kg_m_s"])
    observed_t = traces["time_s"]
    method = report["method_spec"]
    start = method["resolved_start_s"]
    flight = report["flights"][0]  # Chronological selection; no truth-based flight matching.
    takeoff_obs = flight["takeoff_s"]
    takeoff = events["takeoff"]["time_s"]
    landing = events["touchdown"]["time_s"]
    require(
        physics_t[0] <= start < takeoff < landing <= physics_t[-1]
        and observed_t[0] <= start < takeoff_obs <= observed_t[-1] <= physics_t[-1],
        "invalid comparator event order or reference coverage",
    )
    reference_velocity = float(np.interp(takeoff, physics_t, momentum) / mass)
    momentum_start = float(np.interp(start, physics_t, momentum))
    early = (observed_t >= start - 1e-12) & (observed_t <= takeoff_obs + 1e-12)
    observed_impulse = float(
        np.trapezoid(traces["force_n"][early] - report["body_weight_n"], observed_t[early])
    )
    reference_impulse = reference_velocity * mass - momentum_start
    full_impulse = float(np.interp(observed_t[-1], physics_t, momentum) - momentum_start)
    physics_dt = float(verification["reference_resolution_s"])
    event_tolerance = report["uncertainty"]["event_interval_resolution_s"] + 2 * physics_dt
    onset_reference = events["kinetic_onset"]["time_s"]
    quiet = (observed_t >= method["quiet_window_s"][0]) & (observed_t < method["quiet_window_s"][1])
    raw_left = traces["fz_plate1_n"]
    raw_right = traces["fz_plate2_n"]

    def signal(values):
        return {
            "samples": len(values),
            "quiet_mean_n": float(np.mean(values[quiet])),
            "minimum_n": float(np.min(values)),
            "maximum_n": float(np.max(values)),
            "rail_samples": int(np.sum(np.abs(values) >= 5000)),
        }

    return {
        "case_id": case,
        "protocol": protocol,
        "engine_verdict": report["verdict"],
        "diagnostic_only": report["diagnostic_only"],
        "qc_failures": [row for row in report["qc"] if row["status"] == "fail"],
        "uncertainty": report["uncertainty"],
        "force_distribution": {"left": signal(raw_left), "right": signal(raw_right)},
        "additional_detected_flights": report["flights"][1:],
        "reference_resolution_s": physics_dt,
        "clock_reference_check": verification["synchronization_truth_check"],
        "comparisons": {
            "mass_kg": discrepancy(report["mass_kg"], mass, 2 / model["gravity_m_s2"]),
            "body_weight_n": discrepancy(report["body_weight_n"], weight, 2),
            "start_to_takeoff_net_impulse_n_s": discrepancy(
                observed_impulse, reference_impulse, 3.75
            ),
            "whole_record_net_impulse_n_s": discrepancy(
                report["metrics"]["net_impulse_n_s"], full_impulse
            ),
            "takeoff_velocity_m_s": discrepancy(
                flight["takeoff_velocity_m_s"], reference_velocity, 0.05
            ),
            "takeoff_s": discrepancy(takeoff_obs, takeoff, event_tolerance),
            "touchdown_s": discrepancy(flight["landing_s"], landing, event_tolerance),
            "flight_time_s": discrepancy(
                flight["landing_s"] - takeoff_obs if flight["landing_s"] is not None else None,
                landing - takeoff,
                2 * event_tolerance,
            ),
            "onset_s_different_definitions": discrepancy(
                report["onset"]["time_s"], onset_reference
            ),
        },
        "withheld": {
            "all_trial_metrics": "RES-670 trial QC rejected" if report["diagnostic_only"] else None,
            "impact_peak_rfd_and_full_record_agreement": "clipped signals and rigid modeled impact",
            "cop_and_six_axis_wrench": "vertical-only channels",
            "anatomical_com_bar_lpt_metrics": "LPT and body geometry unavailable",
            "complete_measurement_uncertainty": "manufactured, assumed uncalibrated sensors",
        },
    }


def compare(processing, corpus, verification, dataset):
    reports, run_receipt, verified = sealed_run(processing, corpus, verification, dataset)
    cases = []
    for case, protocol in CASES.items():
        run_case = next(c for c in run_receipt["cases"] if c["case_id"] == case)
        verified_case = next(c for c in verified["cases"] if c["case_id"] == case)
        truth = corpus / "TRUTH" / case
        model = load(truth / "reference.json")
        events = load(truth / "model_events.json")["analysis"]["events"]
        hash_manifest = sha256_file(corpus / "SOURCE" / case / "manifest.json")
        hash_truth = sha256_file(truth / "checksums.sha256")
        require(
            model["protocol"] == protocol
            and model["source_manifest_sha256"]
            == run_case["source_manifest_sha256"]
            == hash_manifest
            and run_case["truth_inventory_sha256"] == hash_truth
            and hash_manifest
            == next(f.sha256 for f in dataset.files if f.path == f"{case}/manifest.json")
            and verified_case["synchronization_truth_check"]["status"]
            == "BOUNDED_MAPPING_CONSISTENT_WITH_TRUTH_SAMPLE_TIMES",
            "mismatched trial, TRUTH inventory or reference timing",
        )
        report, traces, report_sha = reports[case]
        case_report = compare_case(case, protocol, report, traces, model, events, verified_case)
        case_report["hashes"] = {
            "source_manifest_sha256": hash_manifest,
            "truth_inventory_sha256": hash_truth,
            "engine_report_sha256": report_sha,
            "trace_sha256": report["trace_output"]["sha256"],
        }
        cases.append(case_report)
    return {
        "schema": "res718-force-comparison@1",
        "scope": "manufactured diagnostic comparison; empirical validity not established",
        "upstream_revision": UPSTREAM,
        "run_receipt_sha256": sha256_file(corpus / "run.json"),
        "corpus_inventory_sha256": sha256_file(corpus / "inventory.json"),
        "post_verifier_sha256": sha256_file(verification),
        "admitted_cases": sum(not c["diagnostic_only"] for c in cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processing", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.processing, args.corpus, args.verification, DatasetManifest.load(args.dataset)
    )
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
