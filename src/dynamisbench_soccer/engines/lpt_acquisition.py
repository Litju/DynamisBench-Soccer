"""Source-scoped cable acquisition and attachment geometry for LPTs and encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.signals.schema import (
    Acquisition,
    AcquisitionSource,
    BundleKind,
    DeviceClass,
    FloatArray,
    Quantity,
    SignalBundle,
)


class CableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source: str
    channel: str
    max_speed_m_s: float = Field(gt=0)
    count_direction: Literal[-1, 1] = 1
    zero_count: int = Field(default=0, gt=-(2**53), lt=2**53)
    count_modulus: int | None = Field(default=None, ge=2, le=2**32)
    anchor_index: int = Field(default=0, ge=0)
    initial_length_m: float | None = Field(default=None, gt=0)
    full_length_m: float | None = Field(default=None, gt=0)
    slack_channel: str | None = None


class GeometrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    attachment_point: str = Field(min_length=1)
    vertical_line_xy_m: tuple[float, float] | None = None
    max_sync_error_s: float | None = Field(default=None, gt=0)
    gap_factor: float = Field(default=3, gt=1)


@dataclass(frozen=True)
class Cable:
    spec: CableSpec
    source: AcquisitionSource
    time_s: FloatArray
    length_m: FloatArray
    absolute: bool
    resolution_m: float | None
    faults: dict[str, list[int]]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class AcquisitionResult:
    time_s: FloatArray
    cables: tuple[Cable, ...]
    attachment_point: str | None
    geometry_method: str
    vertical_displacement_m: FloatArray | None
    limitations: tuple[str, ...]


def _read_cable(bundle: SignalBundle, spec: CableSpec, gap_factor: float) -> Cable:
    acquisition, sampling = bundle.meta.acquisition, bundle.meta.sampling
    assert acquisition is not None and sampling is not None
    source = next((item for item in acquisition.sources if item.id == spec.source), None)
    if source is None or source.device.device_class not in {DeviceClass.LPT, DeviceClass.ENCODER}:
        raise ValueError(f"{spec.source}: expected an LPT or encoder acquisition source")
    if spec.channel not in source.channels:
        raise ValueError(f"{spec.channel}: channel belongs to a different acquisition source")
    cls = source.device.device_class
    column = bundle.meta.channel(spec.channel)
    allowed = (
        {Quantity.COUNT}
        if cls is DeviceClass.ENCODER
        else {Quantity.DISTANCE, Quantity.DISPLACEMENT}
    )
    if column.quantity not in allowed:
        raise ValueError(f"{spec.channel}: expected raw cable distance, displacement or counts")
    times = [
        name for name in source.channels if bundle.meta.channel(name).quantity is Quantity.TIME
    ]
    if len(times) > 1 or (not times and (sampling.irregular or len(acquisition.clocks) > 1)):
        raise ValueError(f"{spec.source}: requires one source-owned native timestamp channel")
    time = (
        bundle[times[0]]
        if times
        else sampling.t0_s + np.arange(bundle.meta.n_rows) / sampling.nominal_hz
    )
    dt = np.diff(time)
    if not len(dt) or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError(f"{spec.source}: requires at least two strictly increasing timestamps")

    records = [
        c
        for c in source.calibration
        if c.channel == spec.channel and (c.gain is not None or c.offset is not None)
    ]
    if len(records) > 1:
        raise ValueError(f"{spec.channel}: ambiguous affine calibration")
    calibration = records[0] if records else None
    resolution = None
    if column.quantity is Quantity.COUNT:
        scale = source.geometry.encoder_counts_per_m
        if "encoder_counts_per_m" in source.geometry.unknown:
            scale = None
        if calibration and (
            calibration.input_unit not in {"1", "count"}
            or calibration.output_unit != "1"
            or calibration.gain != 1
            or calibration.offset not in {0, None}
        ):
            raise ValueError("raw encoder counts need identity calibration and declared resolution")
        if scale is None:
            raise ValueError("missing encoder counts_per_m")
        gain = spec.count_direction / scale
        offset, absolute = 0.0, False
        raw = bundle[spec.channel]
        if np.any(np.abs(raw) >= 2**53) or np.any(raw != np.floor(raw)):
            raise ValueError("encoder needs exactly represented integer counts")
        if spec.count_modulus is not None:
            raw = _unwrap_counts(raw, spec, dt, scale)
        else:
            raw = raw - spec.zero_count
        if np.any(np.abs(raw) >= 2**53):
            raise ValueError("count difference exceeds exact integer range")
        resolution = abs(gain)
    else:
        if spec.count_direction != 1 or spec.zero_count != 0 or spec.count_modulus is not None:
            raise ValueError("count direction, zero count and modulus apply only to an encoder")
        raw = bundle[spec.channel]
        if calibration and (calibration.input_unit, calibration.output_unit) != ("m", "m"):
            raise ValueError("cable calibration requires metre-to-metre units")
        gain = calibration.gain if calibration and calibration.gain is not None else 1.0
        offset = calibration.offset if calibration and calibration.offset is not None else 0.0
        absolute = column.quantity is Quantity.DISTANCE
    if absolute and spec.initial_length_m is not None:
        raise ValueError("initial length is only for relative inputs")
    if spec.initial_length_m is not None:
        offset += spec.initial_length_m
        absolute = True
    length = gain * raw + offset
    if not np.isfinite(length).all() or resolution == 0:
        raise ValueError("invalid calibrated cable values")

    faults = {}
    limitations = [] if times else ["nominal timestamps synthesized"]
    if resolution is None:
        edge = np.diff(length)
        allowance = spec.max_speed_m_s * dt
    else:
        # Compare in counts to keep the one-count allowance exact near a large counter value.
        edge = np.diff(raw) * spec.count_direction
        allowance = spec.max_speed_m_s * dt / resolution + 1
    for flag, mask in (
        ("dropout_gap", dt > gap_factor / sampling.nominal_hz),
        ("count_jump_or_overspeed" if resolution else "overspeed", edge > allowance),
        ("retraction_overspeed", edge < -allowance),
    ):
        if mask.any():
            faults[flag] = (np.flatnonzero(mask) + 1).tolist()
    if spec.slack_channel:
        if (
            spec.slack_channel not in source.channels
            or bundle.meta.channel(spec.slack_channel).quantity is not Quantity.DIMENSIONLESS
            or not np.isin(bundle[spec.slack_channel], [0, 1]).all()
        ):
            raise ValueError("slack indicator must be a source-owned binary channel")
        slack = np.flatnonzero(bundle[spec.slack_channel]).tolist()
        if slack:
            faults["slack"] = slack
    else:
        limitations.append("slack unmonitored; stationary counts cannot identify slack")
    if absolute and np.any(length < 0):
        faults["negative_length"] = np.flatnonzero(length < 0).tolist()
    if spec.full_length_m is not None and absolute:
        saturated = np.flatnonzero(length >= spec.full_length_m - (resolution or 0) / 2).tolist()
        if saturated:
            faults["full_extension"] = saturated
    else:
        limitations.append("full-extension saturation unmonitored")
    return Cable(spec, source, time, length, absolute, resolution, faults, tuple(limitations))


def _unwrap_counts(raw: FloatArray, spec: CableSpec, dt: FloatArray, scale: float) -> FloatArray:
    modulus = spec.count_modulus
    assert modulus is not None
    if not 0 <= spec.zero_count < modulus or np.any((raw < 0) | (raw >= modulus)):
        raise ValueError("encoder counts and zero must lie in [0, count modulus)")
    # With at most one count of quantisation error, the speed bound must identify
    # a unique branch of the modular difference on every native interval.
    if np.any(spec.max_speed_m_s * dt * scale + 1 >= modulus / 2):
        raise ValueError("encoder rollover is ambiguous at the declared speed and sample intervals")
    half = modulus // 2
    if (len(raw) + 1) * modulus >= 2**63:
        raise ValueError("encoder revolution count exceeds integer range")
    changes = np.remainder(np.diff(raw.astype(np.int64)) + np.int64(half), np.int64(modulus))
    initial = np.array([int(raw[0]) - spec.zero_count], dtype=np.int64)
    unwrapped = np.concatenate((initial, changes - np.int64(half))).cumsum(dtype=np.int64)
    if np.any((unwrapped <= -(2**53)) | (unwrapped >= 2**53)):
        raise ValueError("count difference exceeds exact integer range")
    return unwrapped.astype(np.float64)


def _common_time(
    acquisition: Acquisition, cables: tuple[Cable, ...], limit: float | None
) -> FloatArray:
    first = cables[0]
    if len(cables) == 1:
        return first.time_s
    clocks = {item.id: item for item in acquisition.clocks}
    required = {c.source.clock for c in cables}
    if len(required) > 1:
        required.add(acquisition.reference_clock)
    for clock_id in required:
        clock = clocks[clock_id]
        if clock.source == "unknown" or clock.epoch.strip().lower() in {
            "",
            "unknown",
            "unrecorded",
            "not recorded",
        }:
            raise ValueError("dual cables require known clock identity and epoch")
    if first.source.clock == cables[1].source.clock:
        if not np.array_equal(first.time_s, cables[1].time_s):
            raise ValueError("dual cable timestamps differ on their shared clock")
        return first.time_s
    if limit is None:
        raise ValueError("dual cable clocks require an explicit synchronization error budget")
    aligned = []
    uncertainty = 0.0
    for cable in cables:
        t = cable.time_s
        if cable.source.clock != acquisition.reference_clock:
            link = next((a for a in acquisition.alignments if a.clock == cable.source.clock), None)
            if link is None:
                raise ValueError("dual cable clock alignment missing")
            events = [event[0] for event in link.sync_events]
            if min(events) > t[0] + 1e-12 or max(events) < t[-1] - 1e-12:
                raise ValueError("dual cable clock alignment must bracket the observation")
            t = t + link.offset_s + (t - link.origin_s) * link.drift_ppm * 1e-6
            uncertainty += link.uncertainty_s
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("aligned timestamps must be finite and strictly increasing")
        aligned.append(t)
    error = float(np.max(np.abs(aligned[0] - aligned[1]))) + uncertainty
    half_interval = min(float(np.min(np.diff(t))) for t in aligned) / 2
    if error > min(limit, half_interval):
        raise ValueError("dual cable clocks exceed the synchronization error budget")
    return aligned[0]


def _anchor(cable: Cable) -> tuple[float, float, float]:
    geometry = cable.source.geometry
    if (
        "cable_anchors_m" in geometry.unknown
        or cable.spec.anchor_index >= len(geometry.cable_anchors_m)
        or not cable.absolute
    ):
        raise ValueError("anchor reconstruction requires surveyed anchors and absolute lengths")
    return geometry.cable_anchors_m[cable.spec.anchor_index]


def vertical(
    cables: tuple[Cable, ...],
    geometry: GeometrySpec,
    angle_error_rad: float = 0,
    geometry_error_m: float = 0,
) -> tuple[str, FloatArray | None]:
    """Reconstruct attachment displacement under the declared geometric model."""
    first = cables[0]
    if len(cables) == 2:
        left, right = map(_anchor, cables)
        right = (right[0], right[1], right[2] + geometry_error_m)
        dx, dy, dz = np.subtract(right, left)
        if dx <= 0 or abs(dy) > 1e-9:
            raise ValueError("dual anchors must be ordered by x in a vertical x-z plane")
        baseline = np.hypot(dx, dz)
        along = (first.length_m**2 - cables[1].length_m ** 2 + baseline**2) / (2 * baseline)
        height_sq = first.length_m**2 - along**2
        if np.any(height_sq <= 0):
            raise ValueError("dual cable lengths imply no well-defined above-baseline intersection")
        z = left[2] + along * dz / baseline + np.sqrt(height_sq) * dx / baseline
        method = "planar_dual_circle_above_baseline"
    elif geometry.vertical_line_xy_m is not None:
        anchor = _anchor(first)
        xy = np.add(geometry.vertical_line_xy_m, (geometry_error_m, 0))
        horizontal = np.linalg.norm(np.subtract(xy, anchor[:2]))
        height_sq = first.length_m**2 - horizontal**2
        if np.any(height_sq < 0):
            raise ValueError("cable does not intersect the declared vertical line above its anchor")
        z = np.sqrt(height_sq)
        method = "fixed_vertical_line_above_anchor"
    else:
        recorded = first.source.geometry
        if recorded.cable_angle_rad is None or "cable_angle_rad" in recorded.unknown:
            return "cable_only", None
        angle = recorded.cable_angle_rad + angle_error_rad
        if abs(angle) >= np.pi / 2 or np.cos(angle) < 1e-3:
            raise ValueError("angle from upward vertical is singular or outside the model")
        z = (first.length_m - first.length_m[0]) / np.cos(angle)
        method = "constant_cable_angle_vertical_motion"
    if not np.isfinite(z).all():
        raise ValueError("nonfinite geometry reconstruction")
    return method, z - z[0]


def acquire(
    bundle: SignalBundle, specs: tuple[CableSpec, ...], geometry: GeometrySpec
) -> AcquisitionResult:
    meta = bundle.meta
    if (
        meta.data_category is DataCategory.ANALYTICAL
        or "RES-669 stage=analytical_truth" in meta.provenance.transforms
        or meta.kind is not BundleKind.TIMESERIES
        or meta.acquisition is None
        or meta.sampling is None
    ):
        raise ValueError(
            "expected a v2 instrument observation timeseries, excluding analytical references"
        )
    if len(specs) not in {1, 2} or len({(s.source, s.channel) for s in specs}) != len(specs):
        raise ValueError("declare one or two distinct raw cable channels")
    if len(specs) == 2 and geometry.vertical_line_xy_m is not None:
        raise ValueError("fixed vertical line applies only to a single cable")
    cables = tuple(_read_cable(bundle, s, geometry.gap_factor) for s in specs)
    attachment = [
        c.source.geometry.attachment_point
        if "attachment_point" not in c.source.geometry.unknown
        else None
        for c in cables
    ]
    if any(a is not None and a != geometry.attachment_point for a in attachment):
        raise ValueError("acquired attachment point differs from expected attachment point")
    if len(cables) == 2 and None in attachment:
        raise ValueError("dual cables require the same declared attachment point")
    time = _common_time(meta.acquisition, cables, geometry.max_sync_error_s)
    limitations = tuple(item for cable in cables for item in cable.limitations)
    if any(c.faults for c in cables):
        return AcquisitionResult(time, cables, attachment[0], "integrity_failed", None, limitations)
    if not attachment[0]:
        return AcquisitionResult(
            time,
            cables,
            None,
            "cable_only",
            None,
            (
                *limitations,
                "attachment point missing; vertical motion withheld",
            ),
        )
    method, displacement = vertical(cables, geometry)
    if displacement is None:
        limitations += ("geometry missing; vertical motion withheld",)
    return AcquisitionResult(time, cables, attachment[0], method, displacement, limitations)
