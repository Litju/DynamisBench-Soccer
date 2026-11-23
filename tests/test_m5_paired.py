"""Paired analytical method and design checks, with a stdlib endpoint oracle."""

import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from dynamisbench_soccer.agreement.paired_speed import PairedInput, compare

ROOT = Path(__file__).resolve().parents[1] / "benchmark/fixtures/m5-paired"


def _raw(index):
    return (ROOT / f"pair-sequence-{index:02}.json").read_bytes()


def _paired_truth(raw):
    case = json.loads(raw)
    values = []
    for trial in case["trials"]:
        channel = trial["channels"]
        t = channel["time"]
        distance = math.dist(
            (channel["position_x"][0], channel["position_y"][0]),
            (channel["position_x"][-1], channel["position_y"][-1]),
        )
        values.append(distance / (t[-1] - t[0]) - trial["radial_m_s"] * 5 / 3)
    center = statistics.mean(values)
    spread = statistics.stdev(values)
    half = statistics.NormalDist().inv_cdf(0.975) * spread
    return center, spread, (center - half, center + half)


def _common_offset_truth(radial, lps):
    # The four equiprobable method-level sign combinations give the analytical variance.
    shifts = [a * lps - b * radial * 5 / 3 for a in (-1, 1) for b in (-1, 1)]
    return 2 * math.sqrt(statistics.mean(x * x for x in shifts))


@pytest.mark.parametrize(("index", "within"), [(1, True), (2, True), (3, False)])
def test_paired_agreement_matches_independent_endpoint_and_statistics(index, within):
    raw = _raw(index)
    record = PairedInput.model_validate_json(raw)
    result = compare(record)
    expected_bias, expected_sd, expected_limits = _paired_truth(raw)
    assert result.classification == "analytical_model_agreement"
    assert result.agreement.n_pairs == len(record.trials)
    assert result.agreement.bias == pytest.approx(expected_bias, abs=1e-10, rel=0)
    assert result.agreement.sd_difference == pytest.approx(expected_sd, abs=1e-10, rel=0)
    assert result.agreement.limits == pytest.approx(expected_limits, abs=1e-10, rel=0)
    assert result.within_tolerance is within
    if index == 3:
        assert abs(result.agreement.bias) < record.maximum_discrepancy_m_s


@pytest.mark.parametrize("index", [1, 3])
def test_common_calibration_uncertainty_is_separate_from_loa(index):
    record = PairedInput.model_validate_json(_raw(index))
    result = compare(record)
    expected = _common_offset_truth(record.radial_standard_m_s, record.lps_standard_m_s)
    assert result.expanded_uncertainty == pytest.approx(expected, abs=1e-10, rel=0)
    scaled = record.model_copy(update={"radial_standard_m_s": 2 * record.radial_standard_m_s})
    changed = compare(scaled)
    assert changed.agreement == result.agreement
    assert changed.expanded_uncertainty == pytest.approx(
        _common_offset_truth(scaled.radial_standard_m_s, scaled.lps_standard_m_s),
        abs=1e-10,
        rel=0,
    )


@pytest.mark.parametrize(
    ("index", "classification"),
    [
        (4, "definition_mismatch"),
        (5, "failed_signal_qc"),
        (6, "independent_pairs_required"),
        (7, "repeat_sessions_required"),
    ],
)
def test_incompatible_or_invalid_designs_withhold_agreement(index, classification):
    result = compare(PairedInput.model_validate_json(_raw(index)))
    assert result.classification == classification
    assert result.agreement is result.expanded_uncertainty is None
    assert result.pairs == ()


def test_a_relabelled_pair_cannot_increase_the_paired_sample():
    record = PairedInput.model_validate_json(_raw(1))
    first, *rest = record.trials
    duplicate = first.model_copy(update={"id": "alias"})
    trials = (first, *rest, duplicate)
    raw = json.dumps([trial.channels_sha256 for trial in trials], separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    record.meta.provenance.source_sha256 = digest
    record.meta.acquisition.sources[0].source_sha256 = digest
    result = compare(record.model_copy(update={"trials": trials}))
    assert result.classification == "independent_pairs_required"


def test_source_hash_and_frame_are_enforced():
    record = PairedInput.model_validate_json(_raw(1))
    record.trials[0].channels["position_z"][0] += 1
    with pytest.raises(ValueError, match="channel digest"):
        compare(record)
    record = PairedInput.model_validate_json(_raw(1))
    definition = record.declaration.model_copy(update={"lps_z_up_frame": "sensor"})
    with pytest.raises(ValueError, match="Z-up"):
        compare(record.model_copy(update={"declaration": definition}))


def test_paired_grader_requires_the_exact_uncertainty_and_agreement_receipts():
    from dynamisbench_soccer.m4.admission import ReceiptBinding, ReviewedSlice
    from dynamisbench_soccer.m4.custody import AnswerLocation
    from dynamisbench_soccer.m4.grader import _measurement_binding
    from dynamisbench_soccer.m4.measurement import PROCEDURES
    from dynamisbench_soccer.m4.task_registry import fingerprint, load_registry

    registry = Path(__file__).resolve().parents[1] / "benchmark/tasks"
    task = load_registry(registry)["paired-lps-radar-agreement", 1]
    procedure = PROCEDURES[task.methods[0].id]
    placeholder = "0" * 64
    evidence = {
        role: ReceiptBinding(id=f"RCP-gnss_lps-lps-{role}-paired-speed", sha256=placeholder)
        for role in ("I", "L")
    }
    review = ReviewedSlice(
        task_sha256=fingerprint(task),
        source=procedure.source,
        kind=procedure.kind,
        evidence=evidence,
        inapplicable={},
        implementations=tuple(f"src/dynamisbench_soccer/{dep}" for dep in procedure.dependencies),
        oracle=("test-only oracle",),
        files={},
        answer=AnswerLocation(id="test-only", sha256=placeholder),
        private_inputs_sha256=dict.fromkeys(("expected", "oracle", "provenance"), placeholder),
        reviewer="test only",
        review_url="test only",
        review_head="0" * 40,
    )
    assert _measurement_binding(task, review)
    for role, other in (("I", "I-native-bounds"), ("L", "L")):
        wrong = ReceiptBinding(id=f"RCP-gnss_lps-lps-{other}", sha256=placeholder)
        changed = review.model_copy(update={"evidence": evidence | {role: wrong}})
        assert not _measurement_binding(task, changed)
