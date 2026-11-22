"""Compare admission to RES-669 observations and independent analytical geometry."""

from __future__ import annotations

import numpy as np
import pytest

from dynamisbench_soccer.engines.gnss_lps_acquisition import (
    AcquisitionDeclaration,
    AcquisitionRefusal,
    EnuOrigin,
    admit,
)
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import ExportedCase, export_case
from dynamisbench_soccer.reference.faults import PositionFault, PositionKind, inject
from dynamisbench_soccer.signals.schema import (
    CANONICAL_UNIT,
    BundleMeta,
    ChannelSpec,
    Quantity,
    SignalBundle,
)


def track_case(
    kind: str,
    *,
    hz: float = 18,
    latency: float = 0.038,
    drift: float = 160,
    doppler: bool = True,
    raw: ExportedCase | None = None,
) -> tuple[ExportedCase, AcquisitionDeclaration]:
    case = raw or export_case(
        f"gnss_lps.{kind}", hz=hz, latency_s=latency, clock=DeviceClock(0.21, drift)
    )
    meta = case.observation.meta.model_copy(deep=True)
    assert meta.acquisition is not None
    data = {name: values.copy() for name, values in case.observation.data.items()}
    options = dict(
        stage="native",
        native_hz=hz,
        timestamp_basis="receipt",
        latency_s=latency,
        latency_uncertainty_s=0.001,
        bundle_id=meta.bundle_id,
    )

    def add(name: str, quantity: Quantity, value: float) -> None:
        meta.channels.append(
            ChannelSpec(name=name, quantity=quantity, unit=CANONICAL_UNIT[quantity])
        )
        meta.acquisition.sources[0].channels.append(name)
        data[name] = np.full(meta.n_rows, value, dtype=np.float64)

    add("signal_valid", Quantity.DIMENSIONLESS, 1)
    if kind == "gnss":
        options["enu_origin"] = dict(latitude_deg=40, longitude_deg=-3, ellipsoidal_height_m=0)
        add("hdop", Quantity.DIMENSIONLESS, 0.9)
        add("satellites", Quantity.COUNT, 9)
        add("fix_quality", Quantity.COUNT, 4)
        add("multipath", Quantity.DIMENSIONLESS, 0)
        if doppler:
            options["doppler_basis"] = "horizontal"
            add("doppler_speed", Quantity.VELOCITY, 3)
    else:
        options["lps_z_up_frame"] = "field"
        visibility = {anchor: f"visible_{anchor}" for anchor in meta.geometry.lps_anchors_m}
        options["anchor_visibility"] = visibility
        for name in visibility.values():
            add(name, Quantity.DIMENSIONLESS, 1)
    bundle = SignalBundle(BundleMeta.model_validate(meta.model_dump()), data)
    return ExportedCase(case.truth, bundle), AcquisitionDeclaration.model_validate(options)


@pytest.mark.parametrize(("kind", "rate", "drift"), [("gnss", 10, -160), ("lps", 25, 120)])
def test_admit_native_times_clock_latency_and_positions(kind: str, rate: int, drift: float) -> None:
    case, declaration = track_case(kind, hz=rate, drift=drift)
    result = admit(case.observation, declaration)
    truth = case.truth
    np.testing.assert_array_equal(result.clock.device_time_s, case.observation["time"])
    np.testing.assert_allclose(
        result.clock.reference_stamp_s, truth["time"] + declaration.latency_s, atol=1e-12
    )
    np.testing.assert_allclose(result.clock.measurement_time_s, truth["time"], atol=1e-12)
    assert result.clock.observed_device_hz == pytest.approx(rate)
    assert result.clock.observed_reference_hz == pytest.approx(rate / (1 + drift / 1e6))
    np.testing.assert_allclose(
        result.position_m, np.column_stack([truth[f"position_{a}"] for a in "xyz"]), atol=4e-9
    )
    if kind == "gnss":
        np.testing.assert_array_equal(result.doppler_speed_m_s, case.observation["doppler_speed"])
    else:
        assert result.anchors_m == case.observation.meta.geometry.lps_anchors_m
        assert result.doppler_speed_m_s is None


def test_enu_longitude_wrap_and_height_with_closed_form_equator() -> None:
    case, declaration = track_case("gnss", doppler=False)
    bundle = case.observation
    longitude = np.linspace(179.999999, 180.000001, bundle.meta.n_rows)
    bundle["latitude"][:] = 0
    bundle["longitude"][:] = np.where(longitude <= 180, longitude, longitude - 360)
    bundle["height"][:] = 125
    declaration = declaration.model_copy(
        update={
            "enu_origin": EnuOrigin(latitude_deg=0, longitude_deg=180, ellipsoidal_height_m=100)
        }
    )
    result = admit(bundle, declaration)
    offset = np.deg2rad(longitude - 180)
    radius = 6_378_137 + 125
    expected = np.column_stack(
        (radius * np.sin(offset), np.zeros(len(offset)), radius * np.cos(offset) - (radius - 25))
    )
    np.testing.assert_allclose(result.position_m, expected, atol=4e-9)


@pytest.mark.parametrize("kind", ["gnss", "lps"])
@pytest.mark.parametrize(
    ("fault", "message"),
    [
        (PositionFault(PositionKind.DROPOUT, index=10), "dropout"),
        (PositionFault(PositionKind.JUMP, index=10, shift_m=(0, 0, 15)), "position jump"),
        (PositionFault(PositionKind.INTERPOLATED, stride=3), "resampled"),
    ],
)
def test_res669_adversarial_observations(kind: str, fault: PositionFault, message: str) -> None:
    case, declaration = track_case(kind)
    corrupted = inject(case.observation, fault)
    declaration = declaration.model_copy(update={"bundle_id": corrupted.meta.bundle_id})
    with pytest.raises(AcquisitionRefusal, match=message):
        admit(corrupted, declaration)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("satellites", 3, "satellite"),
        ("hdop", 3.01, "HDOP"),
        ("hdop", 0, "HDOP"),
        ("fix_quality", 6, "fix quality"),
        ("multipath", 1, "multipath"),
        ("signal_valid", 0, "signal integrity"),
        ("signal_valid", 0.5, "binary"),
        ("doppler_speed", -1, "Doppler speed"),
    ],
)
def test_single_bad_gnss_sample_refuses(column: str, value: float, message: str) -> None:
    case, declaration = track_case("gnss")
    case.observation[column][9] = value
    with pytest.raises(AcquisitionRefusal, match=message):
        admit(case.observation, declaration)


@pytest.mark.parametrize(
    "fault", ["occlusion", "duplicate", "collinear", "unknown", "missing_flag"]
)
def test_lps_geometry_and_visibility(fault: str) -> None:
    case, declaration = track_case("lps")
    bundle = case.observation
    assert bundle.meta.acquisition is not None
    geometry = bundle.meta.geometry
    if fault == "occlusion":
        bundle["visible_a"][8] = 0
    elif fault == "duplicate":
        geometry.lps_anchors_m["b"] = geometry.lps_anchors_m["a"]
    elif fault == "collinear":
        geometry.lps_anchors_m["c"] = (0, -10, 1)
    elif fault == "unknown":
        geometry.unknown.append("lps_anchors_m")
    elif fault == "missing_flag":
        declaration.anchor_visibility.pop("a")
    bundle.meta.acquisition.sources[0].geometry = geometry
    with pytest.raises(AcquisitionRefusal, match=r"LPS"):
        admit(bundle, declaration)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"native_hz": 10}, "native acquisition rate"),
        ({"stage": "vendor_resampled"}, "resampled"),
        ({"stage": "vendor_processed"}, "processed"),
        ({"latency_uncertainty_s": 0.2}, "uncertainty"),
        ({"timestamp_basis": "measurement"}, "zero latency"),
        ({"maximum_drift_ppm": 10}, "clock drift"),
        ({"bundle_id": "other-bundle"}, "different bundle"),
        ({"enu_origin": None}, "ENU origin"),
        ({"doppler_basis": None}, "Doppler channel"),
    ],
)
def test_contradictory_acquisition_declarations_refuse(change: dict, message: str) -> None:
    case, declaration = track_case("gnss")
    with pytest.raises(AcquisitionRefusal, match=message):
        admit(case.observation, declaration.model_copy(update=change))


def test_implicit_and_duplicate_timestamps_and_analytical_truth_refuse() -> None:
    case, declaration = track_case("lps")
    bundle = case.observation
    assert bundle.meta.acquisition is not None
    bundle.meta.channels.pop(0)
    bundle.meta.acquisition.sources[0].channels.remove("time")
    with pytest.raises(AcquisitionRefusal, match="missing time channel"):
        admit(
            SignalBundle(bundle.meta, {k: v for k, v in bundle.data.items() if k != "time"}),
            declaration,
        )
    case, declaration = track_case("lps")
    case.observation["time"][3] = case.observation["time"][2]
    with pytest.raises(AcquisitionRefusal, match="strictly increasing"):
        admit(case.observation, declaration)
    with pytest.raises(AcquisitionRefusal, match="observation timeseries"):
        admit(case.truth, declaration.model_copy(update={"bundle_id": case.truth.meta.bundle_id}))


def test_drift_needs_bracketing_sync_pulses() -> None:
    case, declaration = track_case("gnss")
    acquisition = case.observation.meta.acquisition
    assert acquisition is not None
    acquisition.alignments[0].sync_events.pop()
    with pytest.raises(AcquisitionRefusal, match="bracketing"):
        admit(case.observation, declaration)


@pytest.mark.parametrize("source_format", ["Vendor Re-Sampled Export", "vendor up-sampled track"])
def test_resampling_provenance_refuses_even_when_rates_agree(source_format: str) -> None:
    case, declaration = track_case("gnss")
    acquisition = case.observation.meta.acquisition
    assert acquisition is not None
    acquisition.sources[0].raw_format = source_format
    with pytest.raises(AcquisitionRefusal, match="resampled"):
        admit(case.observation, declaration)


@pytest.mark.parametrize("frame", [None, "wrong-frame"])
def test_lps_field_axes_require_an_explicit_z_up_frame(frame: str | None) -> None:
    case, declaration = track_case("lps")
    with pytest.raises(AcquisitionRefusal, match="declared Z-up field frame"):
        admit(case.observation, declaration.model_copy(update={"lps_z_up_frame": frame}))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"native_hz": 10}, "native acquisition rate"),
        ({"stage": "vendor_resampled"}, "resampled"),
        ({"stage": "vendor_processed"}, "processed"),
        ({"latency_uncertainty_s": 0.2}, "uncertainty"),
        ({"timestamp_basis": "measurement"}, "zero latency"),
        ({"maximum_drift_ppm": 10}, "clock drift"),
        ({"bundle_id": "other-bundle"}, "different bundle"),
    ],
)
def test_lps_contradictory_native_declarations_are_refused(change: dict, message: str) -> None:
    case, declaration = track_case("lps")
    with pytest.raises(AcquisitionRefusal, match=message):
        admit(case.observation, declaration.model_copy(update=change))


def test_lps_unit_and_resampling_provenance_refusal() -> None:
    case, declaration = track_case("lps")
    acquisition = case.observation.meta.acquisition
    assert acquisition is not None
    source = acquisition.sources[0]
    source.raw_format = "Vendor Re-Sampled Export"
    with pytest.raises(AcquisitionRefusal, match="resampled"):
        admit(case.observation, declaration)
    source.raw_format = "native LPS"
    case.observation.meta.channel("position_x").unit = "cm"
    with pytest.raises(AcquisitionRefusal, match="is not canonical 'm'"):
        admit(case.observation, declaration)


def test_lps_declared_clock_and_latency_against_affine_reference() -> None:
    case, declaration = track_case("lps", hz=25, drift=-160, latency=0.038)
    expected_time = (1 - 160e-6) * np.arange(case.observation.meta.n_rows) / 25
    result = admit(case.observation, declaration)
    np.testing.assert_allclose(result.clock.measurement_time_s, expected_time, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        result.clock.reference_stamp_s, expected_time + 0.038, rtol=0, atol=1e-12
    )
    acquisition = case.observation.meta.acquisition
    assert acquisition is not None
    acquisition.alignments[0].sync_events.pop()
    with pytest.raises(AcquisitionRefusal, match="bracketing"):
        admit(case.observation, declaration)
