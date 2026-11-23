"""Admission of v2 native GNSS fixes and native LPS field position solutions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    BundleMeta,
    DeviceClass,
    FloatArray,
    Quantity,
    SignalBundle,
)


class AcquisitionRefusal(ValueError):
    """The measurement record fails the native position-track contract."""


class EnuOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, strict=True)

    latitude_deg: float = Field(gt=-90, lt=90)
    longitude_deg: float = Field(ge=-180, le=180)
    ellipsoidal_height_m: float = Field(ge=-10_000, le=10_000)


class AcquisitionDeclaration(BaseModel):
    """Producer declarations that the generic v2 bundle cannot express."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, strict=True)

    bundle_id: str
    stage: Literal["native", "vendor_resampled", "vendor_processed"]
    native_hz: float = Field(gt=0)
    timestamp_basis: Literal["measurement", "receipt"]
    latency_s: float = Field(ge=0)
    latency_uncertainty_s: float = Field(ge=0)
    maximum_hdop: float = Field(default=3.0, gt=0)
    enu_origin: EnuOrigin | None = None
    lps_z_up_frame: str | None = Field(default=None, min_length=1)
    anchor_visibility: dict[str, str] = Field(default_factory=dict)
    doppler_basis: Literal["horizontal", "three_dimensional"] | None = None
    minimum_satellites: int = Field(default=4, ge=4)
    maximum_jump_speed_m_s: float = Field(default=25.0, gt=0)
    maximum_rate_error_fraction: float = Field(default=0.03, ge=0, lt=1)
    maximum_jitter_fraction: float = Field(default=0.10, ge=0, lt=0.5)
    maximum_drift_ppm: float = Field(default=1000.0, ge=0, lt=1e6)


@dataclass(frozen=True)
class TrackClock:
    source_clock: str
    reference_clock: str
    device_time_s: FloatArray
    reference_stamp_s: FloatArray
    measurement_time_s: FloatArray
    elapsed_s: FloatArray
    observed_device_hz: float
    observed_reference_hz: float
    jitter_fraction: float
    drift_ppm: float
    alignment_uncertainty_s: float


@dataclass(frozen=True)
class AdmittedTrack:
    device: DeviceClass
    declaration: AcquisitionDeclaration
    source_id: str
    source_format: str
    source_sha256: str | None
    data_sha256: str | None
    provenance: dict[str, object]
    sensor_placement: str | None
    frame_id: str
    frame_definition: str
    anchors_m: dict[str, tuple[float, float, float]]
    clock: TrackClock
    position_m: FloatArray
    doppler_speed_m_s: FloatArray | None


def _channel(bundle: SignalBundle, name: str, quantity: Quantity) -> FloatArray:
    if not any(c.name == name and c.quantity is quantity for c in bundle.meta.channels):
        raise AcquisitionRefusal(f"missing {quantity} channel {name}")
    return bundle[name]


def _flag(bundle: SignalBundle, name: str) -> FloatArray:
    flag = _channel(bundle, name, Quantity.DIMENSIONLESS)
    if not np.all((flag == 0) | (flag == 1)):
        raise AcquisitionRefusal(f"{name} must be binary")
    return flag


def _ecef(fixes: FloatArray) -> FloatArray:
    lat, lon = np.deg2rad(fixes[:, :2]).T
    h = fixes[:, 2]
    flattening = 1 / 298.257223563
    e2 = flattening * (2 - flattening)
    sin_lat = np.sin(lat)
    radius = 6378137.0 / np.sqrt(1 - e2 * sin_lat**2)
    xy = (radius + h) * np.cos(lat)
    return np.column_stack((xy * np.cos(lon), xy * np.sin(lon), (radius * (1 - e2) + h) * sin_lat))


def _gnss_position(bundle: SignalBundle, origin: EnuOrigin) -> FloatArray:
    fixes = np.column_stack(
        (
            _channel(bundle, "latitude", Quantity.LATITUDE),
            _channel(bundle, "longitude", Quantity.LONGITUDE),
            _channel(bundle, "height", Quantity.POSITION),
        )
    )
    if np.any(np.abs(fixes[:, 2]) > 10_000):
        raise AcquisitionRefusal("ellipsoidal heights must be within 10 km of the ellipsoid")
    center = np.array([[origin.latitude_deg, origin.longitude_deg, origin.ellipsoidal_height_m]])
    lat, lon = np.deg2rad(center[0, :2])
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    axes = np.array(
        [[-slon, clon, 0], [-slat * clon, -slat * slon, clat], [clat * clon, clat * slon, slat]]
    )
    return np.asarray((_ecef(fixes) - _ecef(center)) @ axes.T, dtype=np.float64)


def _clock(bundle: SignalBundle, declaration: AcquisitionDeclaration) -> TrackClock:
    meta = bundle.meta
    acquisition, sampling = meta.acquisition, meta.sampling
    assert acquisition is not None and sampling is not None
    source = acquisition.sources[0]
    source_clock = next(c for c in acquisition.clocks if c.id == source.clock)
    reference_clock = next(c for c in acquisition.clocks if c.id == acquisition.reference_clock)
    if any(
        c.source == "unknown" or c.epoch.strip().lower() in {"", "unknown"}
        for c in (source_clock, reference_clock)
    ):
        raise AcquisitionRefusal("source and reference clock origins must be declared")
    time = _channel(bundle, "time", Quantity.TIME)
    if len(time) < 5:
        raise AcquisitionRefusal("at least five native observations are required")
    intervals = np.diff(time)
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
        raise AcquisitionRefusal("native timestamps must be finite and strictly increasing")
    interval = float(np.median(intervals))
    rate = 1 / interval
    rates: tuple[float, ...] = (sampling.nominal_hz, rate)
    if sampling.measured_hz is not None:
        rates += (sampling.measured_hz,)
    if not math.isfinite(rate) or any(
        abs(hz / declaration.native_hz - 1) > declaration.maximum_rate_error_fraction
        for hz in rates
    ):
        raise AcquisitionRefusal("observed or declared rate contradicts native acquisition rate")
    jitter = float(np.max(np.abs(intervals / interval - 1)))
    if jitter > declaration.maximum_jitter_fraction:
        raise AcquisitionRefusal("timestamp dropout or excessive clock jitter")

    drift = uncertainty = offset = 0.0
    if source.clock != acquisition.reference_clock:
        alignment = next(a for a in acquisition.alignments if a.clock == source.clock)
        drift, uncertainty = alignment.drift_ppm, alignment.uncertainty_s
        if abs(drift) > declaration.maximum_drift_ppm or (
            source_clock.drift_ppm is not None
            and not math.isclose(source_clock.drift_ppm, drift, abs_tol=1e-6)
        ):
            raise AcquisitionRefusal("clock drift exceeds limit or conflicts with alignment")
        if drift:
            pulses = [sample for sample, _ in alignment.sync_events]
            if min(pulses) == max(pulses) or (time[0] < min(pulses) or time[-1] > max(pulses)):
                raise AcquisitionRefusal("clock drift requires sync events bracketing the record")
        offset = alignment.offset_s + (float(time[0]) - alignment.origin_s) * drift * 1e-6
    scale = 1 + drift * 1e-6
    if declaration.timestamp_basis == "measurement" and declaration.latency_s != 0:
        raise AcquisitionRefusal("measurement timestamps require zero latency correction")
    if uncertainty + declaration.latency_uncertainty_s >= interval * scale:
        raise AcquisitionRefusal("clock and latency uncertainty reach the native sample interval")
    elapsed = (time - time[0]) * scale
    reference = (float(time[0]) + offset) + elapsed
    measurement = reference - declaration.latency_s
    for values in (elapsed, reference, measurement):
        if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
            raise AcquisitionRefusal("mapped timestamps lose precision or chronology")
    return TrackClock(
        source.clock,
        reference_clock.id,
        time.copy(),
        reference,
        measurement,
        elapsed,
        rate,
        rate / scale,
        jitter,
        drift,
        uncertainty,
    )


def _lps_position(
    bundle: SignalBundle, declaration: AcquisitionDeclaration
) -> tuple[str, str, FloatArray]:
    acquisition = bundle.meta.acquisition
    assert acquisition is not None
    channels = bundle.channels_where(Quantity.POSITION)
    frame_id = channels[0].frame
    frame = next((f for f in acquisition.frames if f.id == frame_id), None)
    if frame is None or frame.kind != "field" or frame.id != declaration.lps_z_up_frame:
        raise AcquisitionRefusal("LPS positions require a declared Z-up field frame")
    anchors = bundle.meta.geometry.lps_anchors_m
    if (
        "lps_anchors_m" in bundle.meta.geometry.unknown
        or len(anchors) < 3
        or any(not name.strip() for name in anchors)
    ):
        raise AcquisitionRefusal("at least three named surveyed LPS anchors are required")
    xyz = np.array(list(anchors.values()))
    if np.any(np.abs(xyz) > 10_000) or len(np.unique(xyz[:, :2], axis=0)) != len(xyz):
        raise AcquisitionRefusal(
            "LPS anchor coordinates must be field-local and horizontally distinct"
        )
    spread = np.linalg.svd(xyz[:, :2] - xyz[:, :2].mean(axis=0), compute_uv=False)
    if spread[1] <= 1e-3 * spread[0]:
        raise AcquisitionRefusal("LPS anchor layout is nearly collinear in the field plane")
    visibility = declaration.anchor_visibility
    if set(visibility) != set(anchors) or len(set(visibility.values())) != len(anchors):
        raise AcquisitionRefusal("distinct visibility channels are required for every LPS anchor")
    for name, column in visibility.items():
        if column == "signal_valid" or np.any(_flag(bundle, column) == 0):
            raise AcquisitionRefusal(f"LPS anchor occlusion or invalid visibility for {name}")
    position = np.column_stack(
        [bundle[c.name] for c in sorted(channels, key=lambda c: c.axis or "")]
    )
    return frame.id, frame.definition, position


def admit(bundle: SignalBundle, declaration: AcquisitionDeclaration) -> AdmittedTrack:
    """Refuse the full record on failed integrity, before any smoothing or derivatives."""
    try:
        declaration = AcquisitionDeclaration.model_validate(declaration.model_dump())
        meta = BundleMeta.model_validate(bundle.meta.model_dump())
        bundle = SignalBundle(meta, bundle.data)
    except ValueError as exc:
        raise AcquisitionRefusal(str(exc)) from exc
    if declaration.bundle_id != meta.bundle_id:
        raise AcquisitionRefusal("acquisition declaration belongs to a different bundle")
    device = meta.device.device_class
    if (
        meta.data_category not in {DataCategory.SYNTHETIC, DataCategory.EMPIRICAL}
        or device not in {DeviceClass.GNSS, DeviceClass.LPS}
        or meta.kind is not BundleKind.TIMESERIES
    ):
        raise AcquisitionRefusal("native GNSS or LPS observation timeseries required")
    acquisition = meta.acquisition
    if acquisition is None or len(acquisition.sources) != 1:
        raise AcquisitionRefusal("one explicit GNSS or LPS acquisition source is required")
    source = acquisition.sources[0]
    if source.device != meta.device or source.geometry != meta.geometry:
        raise AcquisitionRefusal("source and trial device or geometry disagree")
    provenance = [source.raw_format, meta.provenance.created_by, *meta.provenance.transforms]
    markers = (
        "vendorprocessed",
        "manufacturerprocessed",
        "resampl",
        "upsampl",
        "interpolat",
        "downsampl",
        "decimat",
        "vendorsmoothed",
        "manufacturersmoothed",
    )
    normalized = [re.sub(r"[^a-z0-9]+", "", item.lower()) for item in provenance]
    if declaration.stage != "native" or any(
        marker in item for item in normalized for marker in markers
    ):
        raise AcquisitionRefusal("vendor-processed or resampled samples cannot enter as native")
    clock = _clock(bundle, declaration)
    if np.any(_flag(bundle, "signal_valid") == 0):
        raise AcquisitionRefusal("signal integrity flag indicates loss or invalid solution")

    doppler = None
    anchors: dict[str, tuple[float, float, float]] = {}
    if device is DeviceClass.GNSS:
        origin = declaration.enu_origin
        if (
            origin is None
            or meta.geometry.geodetic_datum != "WGS84"
            or ("geodetic_datum" in meta.geometry.unknown)
        ):
            raise AcquisitionRefusal("WGS84 fixes and an explicit ENU origin are required")
        if declaration.anchor_visibility or declaration.lps_z_up_frame is not None:
            raise AcquisitionRefusal("anchor visibility and Z-up field declarations are LPS only")
        fix_frames = {
            meta.channel(c).frame for c in ("latitude", "longitude", "height") if c in bundle.data
        }
        if len(fix_frames) != 1 or not any(
            f.id in fix_frames and f.kind == "geodetic" for f in acquisition.frames
        ):
            raise AcquisitionRefusal("latitude, longitude and height must share a geodetic frame")
        hdop = _channel(bundle, "hdop", Quantity.DIMENSIONLESS)
        if np.any((hdop <= 0) | (hdop > declaration.maximum_hdop)):
            raise AcquisitionRefusal("HDOP outside declared quality limit")
        if np.any(_channel(bundle, "satellites", Quantity.COUNT) < declaration.minimum_satellites):
            raise AcquisitionRefusal("low satellite count")
        if not np.all(np.isin(_channel(bundle, "fix_quality", Quantity.COUNT), [1, 2, 4, 5])):
            raise AcquisitionRefusal("unsupported or invalid GNSS fix quality")
        if np.any(_flag(bundle, "multipath") != 0):
            raise AcquisitionRefusal("GNSS multipath flagged")
        if ("doppler_speed" in bundle.data) != (declaration.doppler_basis is not None):
            raise AcquisitionRefusal(
                "Doppler channel and declared speed basis must accompany each other"
            )
        if declaration.doppler_basis is not None:
            doppler = _channel(bundle, "doppler_speed", Quantity.VELOCITY).copy()
            if np.any((doppler < 0) | (doppler > declaration.maximum_jump_speed_m_s)):
                raise AcquisitionRefusal("Doppler speed outside declared motion limit")
        position = _gnss_position(bundle, origin)
        frame_id, definition = "local_enu", "WGS84 tangent frame: x east, y north, z up"
    else:
        if (
            declaration.enu_origin is not None
            or declaration.doppler_basis is not None
            or ("doppler_speed" in bundle.data)
        ):
            raise AcquisitionRefusal(
                "LPS uses its declared field frame and has no GNSS Doppler speed"
            )
        frame_id, definition, position = _lps_position(bundle, declaration)
        anchors = meta.geometry.lps_anchors_m.copy()
    if np.any(np.abs(position) > 10_000):
        raise AcquisitionRefusal("positions must be within 10 km per axis of the local origin")
    step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    if np.any(step > declaration.maximum_jump_speed_m_s * np.diff(clock.elapsed_s)):
        raise AcquisitionRefusal("position jump exceeds declared 3D step-speed limit")
    return AdmittedTrack(
        device,
        declaration,
        source.id,
        source.raw_format,
        source.source_sha256,
        meta.data_sha256,
        meta.provenance.model_dump(mode="json"),
        meta.geometry.sensor_placement,
        frame_id,
        definition,
        anchors,
        clock,
        position,
        doppler,
    )
