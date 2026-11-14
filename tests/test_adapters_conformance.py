"""Adapter conformance suite (RES-667). Every adapter must pass on Linux CI."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.adapters.base import AdapterError, all_adapters, get
from dynamisbench_soccer.signals import examples, io
from dynamisbench_soccer.signals.schema import CANONICAL_UNIT, BundleKind

FIX = Path(__file__).parent / "fixtures" / "adapters"

VALID = {
    "forceplate_dual_N.csv": "generic_forceplate_csv",
    "forceplate_single_kgf_time.csv": "generic_forceplate_csv",
    "nhe_kgf.csv": "generic_dynamometer_csv",
    "hip_squeeze_45.csv": "generic_dynamometer_csv",
    "gates_ms.csv": "timing_gates_csv",
    "radar_kmh_time.csv": "radar_laser_csv",
    "lpt_mm.csv": "lpt_csv",
    "gnss_summary.csv": "gnss_summary_csv",
}

INVALID = {
    "fp_missing_sampling_rate.csv": ("generic_forceplate_csv", "missing sampling rate"),
    "fp_unit_mislabel_force_in_m.csv": ("generic_forceplate_csv", "unit mislabel"),
    "fp_unknown_unit.csv": ("generic_forceplate_csv", "unknown unit"),
    "fp_no_unit.csv": ("generic_forceplate_csv", r"name\[unit\]"),
    "fp_non_numeric.csv": ("generic_forceplate_csv", "non-numeric"),
    "fp_ragged_row.csv": ("generic_forceplate_csv", "expected 2 fields"),
    "fp_time_not_monotonic.csv": ("generic_forceplate_csv", "strictly increasing"),
    "fp_missing_magic.csv": ("generic_forceplate_csv", "dbs-generic-csv"),
    "fp_unknown_metadata_key.csv": ("generic_forceplate_csv", "unknown metadata keys"),
    "fp_missing_subject.csv": ("generic_forceplate_csv", "subject"),
    "fp_wrong_test_for_adapter.csv": ("generic_forceplate_csv", "does not support test_id"),
    "fp_bad_data_category.csv": ("generic_forceplate_csv", "data_category"),
    "fp_no_data_rows.csv": ("generic_forceplate_csv", "no data rows"),
    "fp_duplicate_columns.csv": ("generic_forceplate_csv", "duplicate column"),
    "fp_nan_value.csv": ("generic_forceplate_csv", "non-finite"),
    "dyn_hip_missing_angle.csv": ("generic_dynamometer_csv", "hip_flexion_deg"),
    "dyn_hip_missing_action.csv": ("generic_dynamometer_csv", "action"),
    "dyn_force_in_seconds.csv": ("generic_dynamometer_csv", "unit mislabel"),
    "gates_with_sampling_rate.csv": ("timing_gates_csv", "sampling_rate_hz is not allowed"),
    "gates_time_in_metres.csv": ("timing_gates_csv", "unit mislabel"),
    "radar_velocity_in_newtons.csv": ("radar_laser_csv", "unit mislabel"),
    "lpt_extra_column.csv": ("lpt_csv", "exactly 'displacement"),
}


def test_every_adapter_has_valid_fixture() -> None:
    assert set(VALID.values()) == set(all_adapters())


def test_every_invalid_fixture_is_listed() -> None:
    assert {p.name for p in (FIX / "invalid").glob("*.csv")} == set(INVALID)
    assert {p.name for p in (FIX / "valid").glob("*.csv")} == set(VALID)


@pytest.mark.parametrize(("fname", "adapter"), sorted(VALID.items()))
def test_valid_export_converts_to_canonical_bundle(
    fname: str, adapter: str, tmp_path: Path
) -> None:
    path = FIX / "valid" / fname
    bundle = get(adapter).read(path)
    meta = bundle.meta
    cap = get(adapter).capabilities
    assert meta.kind is cap.kind
    assert meta.device.device_class in cap.device_classes
    assert meta.test_id in cap.test_ids
    assert meta.provenance.created_by == f"adapter:{cap.name}@{cap.version}"
    assert meta.provenance.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    for ch in meta.channels:
        assert ch.unit == CANONICAL_UNIT[ch.quantity]
    if meta.kind is BundleKind.TIMESERIES:
        assert meta.sampling is not None
    else:
        assert meta.sampling is None
    # the bundle must survive the canonical writer/reader unchanged
    io.write(bundle, tmp_path / "b")
    again = io.read(tmp_path / "b")
    assert again.meta == bundle.meta


@pytest.mark.parametrize(("fname", "case"), sorted(INVALID.items()))
def test_malformed_exports_are_rejected(fname: str, case: tuple[str, str]) -> None:
    adapter, message = case
    with pytest.raises(AdapterError, match=message):
        get(adapter).read(FIX / "invalid" / fname)


def test_unit_conversions_are_applied_and_recorded() -> None:
    fp = get("generic_forceplate_csv").read(FIX / "valid" / "forceplate_single_kgf_time.csv")
    assert any("kgf" in t for t in fp.meta.provenance.transforms)
    assert fp["fz"][0] == pytest.approx(75.0 * 9.80665, rel=1e-4)
    gates = get("timing_gates_csv").read(FIX / "valid" / "gates_ms.csv")
    np.testing.assert_allclose(gates["t_s"], [0.0, 1.05, 1.80, 3.05, 4.22])
    radar = get("radar_laser_csv").read(FIX / "valid" / "radar_kmh_time.csv")
    expected = examples.sprint_radar(fs=47.0)["velocity"]
    np.testing.assert_allclose(radar["velocity"], expected, atol=1e-5)
    lpt = get("lpt_csv").read(FIX / "valid" / "lpt_mm.csv")
    assert lpt["displacement"].max() == pytest.approx(0.35, abs=1e-3)


def test_sampling_inferred_from_time_column_is_recorded() -> None:
    fp = get("generic_forceplate_csv").read(FIX / "valid" / "forceplate_single_kgf_time.csv")
    assert fp.meta.sampling is not None
    assert fp.meta.sampling.nominal_hz == pytest.approx(200.0)
    assert any("inferred from time column" in t for t in fp.meta.provenance.transforms)


def test_vendor_values_go_to_separate_namespace() -> None:
    fp = get("generic_forceplate_csv").read(FIX / "valid" / "forceplate_dual_N.csv")
    assert fp.meta.vendor_reported["jump_height"].value == pytest.approx(0.312)
    assert "jump_height" not in fp.data


def test_dual_plate_mapping_and_summed_policy() -> None:
    fp = get("generic_forceplate_csv").read(FIX / "valid" / "forceplate_dual_N.csv")
    sides = {c.name: (c.side, c.plate) for c in fp.meta.channels}
    assert sides == {"fz_left": ("left", 1), "fz_right": ("right", 2)}
    assert fp.meta.geometry.summed_force_policy == "sum_fz_all_plates"
