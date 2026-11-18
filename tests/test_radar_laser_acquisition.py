"""RES-699 admission controls against independent RES-669 observations."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines.radar_laser_acquisition import (
    AcquisitionRefusal,
    ChannelStage,
    admit,
)
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle


@pytest.mark.parametrize("family,rate", [("radar", 47), ("laser", 256)])
def test_admits_distinct_native_measurands_and_affine_clocks(
    family: str, rate: int
) -> None:
    case = export_case(
        f"radar_laser.{family}", hz=rate, clock=DeviceClock(0.13, 90)
    )
    trace = admit(case.observation, stage=ChannelStage.NATIVE)
    assert trace.channel == ("velocity" if family == "radar" else "distance")
    assert trace.measured_hz == pytest.approx(rate, rel=1e-10)
    assert trace.native_nominal_hz == rate
    np.testing.assert_allclose(
        trace.reference_time_s, case.truth["time"], atol=1e-12
    )
    assert trace.cosine_angle_rad == (pytest.approx(0.2) if family == "radar" else None)
    assert trace.aiming_point
    assert trace.n_samples == case.observation.meta.n_rows


def _edit(bundle: SignalBundle, edit) -> SignalBundle:  # type: ignore[no-untyped-def]
    raw = bundle.meta.model_dump(mode="json")
    data = {key: value.copy() for key, value in bundle.data.items()}
    edit(raw, data)
    return SignalBundle(BundleMeta.model_validate(raw), data)


def test_rejects_truth_wrong_device_and_undeclared_processing_stage() -> None:
    radar = export_case("radar_laser.radar")
    with pytest.raises(AcquisitionRefusal, match="truth"):
        admit(radar.truth, stage=ChannelStage.NATIVE)
    with pytest.raises(AcquisitionRefusal, match="stage"):
        admit(radar.observation, stage="native")  # type: ignore[arg-type]
    with pytest.raises(AcquisitionRefusal, match="GNSS speed"):
        admit(export_case("gnss_lps.gnss").observation, stage=ChannelStage.NATIVE)


def test_processed_data_are_admitted_only_under_honest_stage() -> None:
    radar = export_case("radar_laser.radar").observation

    def processed(raw, data) -> None:  # type: ignore[no-untyped-def]
        raw["acquisition"]["sources"][0]["raw_format"] = "vendor_processed_resampled"

    data = _edit(radar, processed)
    with pytest.raises(AcquisitionRefusal, match="processed/resampled"):
        admit(data, stage=ChannelStage.NATIVE)
    processed_trace = admit(data, stage=ChannelStage.MANUFACTURER_PROCESSED)
    assert any("opaque" in w for w in processed_trace.warnings)


def test_unmeasured_geometry_cannot_be_inferred() -> None:
    radar = export_case("radar_laser.radar").observation

    def unknown(raw, data) -> None:  # type: ignore[no-untyped-def]
        raw["acquisition"]["sources"][0]["geometry"]["aiming_point"] = None
        raw["acquisition"]["sources"][0]["geometry"]["unknown"].append("aiming_point")

    with pytest.raises(AcquisitionRefusal, match="aiming point"):
        admit(_edit(radar, unknown), stage=ChannelStage.NATIVE)


def test_native_timestamp_fault_and_dropouts_refuse() -> None:
    laser = export_case("radar_laser.laser", hz=100).observation

    def backstep(raw, data) -> None:  # type: ignore[no-untyped-def]
        data["time"][5] = data["time"][4]

    with pytest.raises(AcquisitionRefusal, match="non-monotonic"):
        admit(_edit(laser, backstep), stage=ChannelStage.NATIVE)
    missing = inject(
        laser, ContinuousFault(ContinuousKind.DROP, ("distance",), stride=5)
    )
    with pytest.raises(AcquisitionRefusal, match="timestamp gap"):
        admit(missing, stage=ChannelStage.NATIVE)


def test_source_calibration_ambiguity_refuses() -> None:
    radar = export_case("radar_laser.radar").observation

    def contradictory(raw, data) -> None:  # type: ignore[no-untyped-def]
        other = raw["acquisition"]["sources"][0]["calibration"][0].copy()
        other["cosine_angle_rad"] = 0.5
        raw["acquisition"]["sources"][0]["calibration"].append(other)

    with pytest.raises(AcquisitionRefusal, match="contradictory"):
        admit(_edit(radar, contradictory), stage=ChannelStage.NATIVE)


def test_engine_is_independent_of_reference_generators() -> None:
    root = Path(__file__).resolve().parents[1]
    module = root / "src/dynamisbench_soccer/engines/radar_laser_acquisition.py"
    imports = ast.walk(ast.parse(module.read_text()))
    assert not any(
        "dynamisbench_soccer.reference" in ast.unparse(n)
        for n in imports if isinstance(n, (ast.Import, ast.ImportFrom))
    )
