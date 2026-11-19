import hashlib
import json

import numpy as np
import pytest

from dynamisbench_soccer.adapters.biosyngen_force import CASES, DATASET, UPSTREAM, convert
from dynamisbench_soccer.data.manifest import DatasetManifest, describe_directory
from dynamisbench_soccer.engines.force_platform_acquisition import acquire

CASE, OTHER = CASES
ASSUMED = "ASSUMED_MANUFACTURED_PARAMETERS_NOT_EMPIRICAL_CALIBRATION"


def dump(path, obj):
    path.write_text(json.dumps(obj))


def read(path):
    return json.loads(path.read_text())


def seal_case(root):
    source = read(root / "manifest.json")
    notes = json.loads(source["operator_notes"])
    capture = notes["sync_capture"]
    capture["recording_hashes"]["force"] = hashlib.sha256(
        (root / "streams/force.csv").read_bytes()
    ).hexdigest()
    source["operator_notes"] = json.dumps(notes)
    dump(root / "manifest.json", source)
    dump(
        root / "capture_authority.json",
        {
            **capture,
            "source_manifest_sha256": hashlib.sha256(
                (root / "manifest.json").read_bytes()
            ).hexdigest(),
        },
    )


def trusted(root):
    return DatasetManifest.model_validate(
        {
            "dataset_id": DATASET,
            "version": UPSTREAM,
            "title": "test doubles",
            "data_category": "synthetic",
            "usage": "cross-validation-only",
            "license": "test",
            "redistribution_allowed": False,
            "source": {
                "kind": "generated",
                "location": f"https://github.com/Litju/BioSynGenBenchmark/tree/{UPSTREAM}",
            },
            "files": describe_directory(root),
        }
    )


@pytest.fixture
def corpus(tmp_path):
    for trial_index, case in enumerate(CASES):
        folder = tmp_path / case
        (folder / "streams").mkdir(parents=True)
        (folder / "streams/force.csv").write_text(
            "sample_index,time_s,force_left_N,force_right_N\n"
            f"0,0.002,{10 + trial_index},25\n1,0.003,20,15\n2,0.004,30,35\n"
        )
        captures = {
            "contract_id": "BSMB-FOUR-EDGE-CAPTURE-1.0",
            "reference_clock_id": "external-pulse-clock",
            "capture_error_bound_s": 2e-6,
            "devices": {
                k: {"clock_id": k, "edge_native_times_s": [0, 0.1, 0.2, 0.3]}
                for k in ("force", "lpt")
            },
            "fit_edge_indices": [0, 3],
            "held_out_edge_indices": [1, 2],
            "recording_hashes": {},
            "reference_edge_times_s": [0, 0.1, 0.2, 0.3],
        }
        mapping = {k: {"offset_s": 0, "drift_scale": 1} for k in ("force", "lpt")}
        dump(
            folder / "source-verification.json",
            {
                "source": "PASS",
                "synchronization": "BOUNDED_CAPTURE_MAPPING",
                "clock_admission_contract": "BSMB-CLOCK-ADMISSION-1.0",
                "nominal_rates_hz": {"force": 1000, "lpt": 1000},
                "declared_jitter_bound_s": 0,
                "dropped_sample_counts": {"force": 0},
                "stream_mappings": mapping,
                "worst_case_mapping_error_s": {"force": 2e-5},
            },
        )
        dump(
            folder / "manifest.json",
            {
                **dict.fromkeys(("acquisition_id", "bundle_id", "trial_id"), case),
                "created_at_utc": "2026-10-15T00:00:00Z",
                "profile_id": "BSMB-PAIRED-CABLE-ANGLE-1.1",
                "profile_version": "1.1",
                "synchronization": {"stream_mappings": mapping},
                "force_stream": {
                    "path": "streams/force.csv",
                    "columns": ["sample_index", "time_s", "force_left_N", "force_right_N"],
                    "unit": "N",
                    "frame_id": "LAB_VERTICAL_Z_UP",
                    "positive_direction": "UP",
                    "processing": {"state": "RAW_UNFILTERED_DECLARED"},
                    "calibration": {"evidence_reference": ASSUMED},
                    "timebase": {
                        "clock_id": "force",
                        "authority": "TIMESTAMP",
                        "nominal_sampling_rate_hz": 1000,
                    },
                    "left_plate": {"manufacturer": "test", "model": "dual"},
                },
                "operator_notes": json.dumps(
                    {
                        "generator_contract_id": "BSMB-GENERATOR-1.1",
                        "acquisition_profile": "BSMB-FORCE1000-LPT1000-1.0",
                        "sync_capture": captures,
                        "force_front_end": {
                            "adc_range": [2500, 5000, 1500],
                            **{side: {"gain": np.eye(3).tolist()} for side in ("left", "right")},
                        },
                    }
                ),
            },
        )
        seal_case(folder)
    return tmp_path


def test_v2_retains_independent_plates_and_shared_native_clock(corpus):
    bundle = convert(corpus / CASE, trusted(corpus))
    assert bundle.meta.test_id == "cmj"
    assert bundle.meta.acquisition.sources[0].calibration_unknown_reason == "ASSUMED_UNCALIBRATED"
    assert [p.side for p in acquire(bundle).plates] == ["left", "right"]
    np.testing.assert_array_equal(bundle["time"], [0.002, 0.003, 0.004])
    np.testing.assert_array_equal(bundle["fz_left"], [10, 20, 30])
    np.testing.assert_array_equal(bundle["fz_right"], [25, 15, 35])
    np.testing.assert_array_equal(acquire(bundle).vertical_force_n, [35, 35, 65])


@pytest.mark.parametrize("mutation", ["swap", "corrupt", "other_trial", "truth", "linked_truth"])
def test_pinned_source_rejects_mutation(corpus, mutation):
    expected = trusted(corpus)
    root = corpus / CASE
    force = root / "streams/force.csv"
    content = force.read_text()
    match = "SHA-256"
    if mutation == "swap":
        force.write_text(content.replace("10,25", "25,10"))
    elif mutation == "corrupt":
        force.write_text(content.replace("30,35", "30,37"))
    elif mutation == "other_trial":
        force.write_bytes((corpus / OTHER / "streams/force.csv").read_bytes())
        seal_case(root)
        match = "SHA-256|size mismatch"
    elif mutation == "truth":
        (root / "reference.json").write_text("{}")
        match = "SOURCE-only"
    else:
        truth = corpus / "truth.csv"
        truth.write_text(content)
        force.unlink()
        force.hardlink_to(truth)
        match = "unsafe or linked"
    with pytest.raises(ValueError, match=match):
        convert(root, expected)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("units", "vertical FORCE in N"),
        ("gain", "gain outside"),
        ("clock", "invalid native FORCE"),
        ("unsynchronized", "clock capture authority"),
        ("nonmonotonic", "strictly native"),
        ("drop", "missing, duplicated"),
        ("trial", "trial identity"),
        ("mappings", "verified clock mappings differ"),
    ],
)
def test_semantic_falsifiers_even_after_test_only_rehash(corpus, change, match):
    root = corpus / CASE
    source = read(root / "manifest.json")
    if change == "units":
        source["force_stream"]["unit"] = "kgf"
    elif change == "gain":
        notes = json.loads(source["operator_notes"])
        notes["force_front_end"]["left"]["gain"][1][1] = 9
        source["operator_notes"] = json.dumps(notes)
    elif change == "clock":
        source["force_stream"]["timebase"]["clock_id"] = "lpt"
    elif change == "trial":
        source["trial_id"] = OTHER
    elif change == "mappings":
        source["synchronization"]["stream_mappings"]["force"]["offset_s"] = 0.1
    elif change == "unsynchronized":
        receipt = read(root / "source-verification.json")
        receipt["synchronization"] = "REFUSED_UNBOUNDED_SAMPLE_JITTER"
        dump(root / "source-verification.json", receipt)
    else:
        force = root / "streams/force.csv"
        old = force.read_text()
        force.write_text(
            old.replace("1,0.003", "1,0.002" if change == "nonmonotonic" else "2,0.003")
        )
    dump(root / "manifest.json", source)
    seal_case(root)
    with pytest.raises(ValueError, match=match):
        convert(root, trusted(corpus))
