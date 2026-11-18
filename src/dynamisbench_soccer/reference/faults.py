"""Typed deterministic observation corruptions; analytical truth is never accepted.

Parameters use each affected channel's canonical units, except explicit metre-vector
position jumps. Outputs remain schema-v2 readable so QC can inspect injected defects.
Every operation records its complete parameters/seed and a canonical-content parent.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Literal

import numpy as np

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.reference.coordinates import LocalFrame
from dynamisbench_soccer.reference.exporters import CORRUPTED_STAGE, OBSERVATION_STAGE, content_hash
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    BundleMeta,
    DeviceClass,
    FloatArray,
    Frame,
    Parent,
    Quantity,
    Side,
    SignalBundle,
)


class ContinuousKind(StrEnum):
    NOISE = "noise"
    OFFSET = "offset"
    DRIFT = "drift"
    CLIPPING = "clipping"
    RESAMPLE = "resample_decimate"
    DROP = "drop_samples"


@dataclass(frozen=True)
class ContinuousFault:
    kind: ContinuousKind
    channels: tuple[str, ...]
    magnitude: float = 0
    seed: int = 0
    stride: int = 2


class EventKind(StrEnum):
    MISSING = "missing_event"
    DUPLICATE = "duplicate_event"
    OUT_OF_ORDER = "out_of_order_event"
    LATENCY = "trigger_latency"


@dataclass(frozen=True)
class EventFault:
    kind: EventKind
    index: int = 1
    latency_s: float = 0.02


class PositionKind(StrEnum):
    NOISE = "position_noise"
    DROPOUT = "position_dropout"
    JUMP = "position_jump"
    INTERPOLATED = "position_interpolated_rate"


@dataclass(frozen=True)
class PositionFault:
    kind: PositionKind
    index: int = 1
    count: int = 1
    shift_m: tuple[float, float, float] = (1, 0, 0)
    noise_sd_m: float = 0.02
    seed: int = 0
    stride: int = 5


class InertialKind(StrEnum):
    BIAS = "inertial_bias"
    MISALIGNMENT = "inertial_misalignment"
    SATURATION = "inertial_saturation"


@dataclass(frozen=True)
class InertialFault:
    kind: InertialKind
    quantity: Literal["specific_force", "angular_velocity"] = "specific_force"
    bias: tuple[float, float, float] = (0, 0, 0)
    rotation: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    limit: float = 5


class ProtocolKind(StrEnum):
    QUIET_MOTION = "quiet_stance_motion"
    DOUBLE_CONTACT = "double_contact"
    STEP_OFF = "step_off"
    SIDE_SWAP = "left_right_swap"


@dataclass(frozen=True)
class ProtocolFault:
    kind: ProtocolKind
    index: int = 0
    magnitude_n: float = 50
    window_s: float = 0.05


Fault = ContinuousFault | EventFault | PositionFault | InertialFault | ProtocolFault


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _index(index: int, n: int) -> None:
    _integer(index, "index")
    if index >= n:
        raise ValueError("fault index outside observation")


def _rows(data: dict[str, FloatArray], indices: np.ndarray) -> dict[str, FloatArray]:
    return {name: values[indices].copy() for name, values in data.items()}


def _continuous(
    bundle: SignalBundle, data: dict[str, FloatArray], fault: ContinuousFault
) -> dict[str, FloatArray]:
    if not isinstance(fault.kind, ContinuousKind) or bundle.meta.kind is not BundleKind.TIMESERIES:
        raise ValueError("continuous fault requires timeseries and ContinuousKind")
    device = bundle.meta.device.device_class
    if device in {DeviceClass.ENCODER, DeviceClass.GNSS, DeviceClass.LPS, DeviceClass.IMU}:
        raise ValueError("use the device's event/position/inertial fault paradigm")
    if not fault.channels or len(set(fault.channels)) != len(fault.channels):
        raise ValueError("unique affected channels required")
    for name in fault.channels:
        if name not in data or bundle.meta.channel(name).quantity in {
            Quantity.TIME,
            Quantity.COUNT,
            Quantity.DIMENSIONLESS,
        }:
            raise ValueError("continuous faults require existing continuous measurement channels")
    if not math.isfinite(fault.magnitude):
        raise ValueError("fault magnitude must be finite")
    _integer(fault.seed, "seed")
    _integer(fault.stride, "stride", 2)
    times = data["time"]
    if fault.kind is ContinuousKind.RESAMPLE:
        return _rows(data, np.arange(0, len(times), fault.stride))
    if fault.kind is ContinuousKind.DROP:
        keep = np.ones(len(times), dtype=bool)
        keep[1 :: fault.stride] = False
        return _rows(data, np.flatnonzero(keep))
    if fault.kind in {ContinuousKind.NOISE, ContinuousKind.CLIPPING} and fault.magnitude <= 0:
        raise ValueError("noise SD and clipping limits must be positive")
    rng = np.random.default_rng(fault.seed)
    for name in fault.channels:
        if fault.kind is ContinuousKind.NOISE:
            data[name] += rng.normal(0, fault.magnitude, len(times))
        elif fault.kind is ContinuousKind.OFFSET:
            data[name] += fault.magnitude
        elif fault.kind is ContinuousKind.DRIFT:
            data[name] += fault.magnitude * (times - times[0])
        elif fault.kind is ContinuousKind.CLIPPING:
            data[name] = np.clip(data[name], -fault.magnitude, fault.magnitude)
    return data


def _event(
    bundle: SignalBundle, data: dict[str, FloatArray], fault: EventFault
) -> dict[str, FloatArray]:
    if not isinstance(fault.kind, EventKind) or (
        bundle.meta.kind is not BundleKind.EVENTS
        and bundle.meta.device.device_class is not DeviceClass.ENCODER
    ):
        raise ValueError("event fault requires events/count stream and EventKind")
    n = bundle.meta.n_rows
    _index(fault.index, n)
    if not math.isfinite(fault.latency_s) or fault.latency_s < 0:
        raise ValueError("nonnegative finite event latency required")
    indices = np.arange(n)
    if fault.kind is EventKind.MISSING:
        if n < 2:
            raise ValueError("cannot remove the only event")
        return _rows(data, np.delete(indices, fault.index))
    if fault.kind is EventKind.DUPLICATE:
        return _rows(data, np.insert(indices, fault.index, fault.index))
    if fault.kind is EventKind.OUT_OF_ORDER:
        if fault.index + 1 >= n:
            raise ValueError("out-of-order fault requires a subsequent event")
        indices[fault.index : fault.index + 2] = indices[fault.index : fault.index + 2][::-1]
        return _rows(data, indices)
    timestamp = "t_s" if bundle.meta.kind is BundleKind.EVENTS else "time"
    data[timestamp][fault.index] += fault.latency_s
    return data


def _position(
    bundle: SignalBundle, data: dict[str, FloatArray], fault: PositionFault
) -> dict[str, FloatArray]:
    if not isinstance(fault.kind, PositionKind) or bundle.meta.device.device_class not in {
        DeviceClass.GNSS,
        DeviceClass.LPS,
    }:
        raise ValueError("position fault requires GNSS/LPS and PositionKind")
    n = bundle.meta.n_rows
    _index(fault.index, n)
    _integer(fault.count, "count", 1)
    _integer(fault.stride, "stride", 2)
    if fault.index + fault.count > n:
        raise ValueError("position fault range outside observation")
    if len(fault.shift_m) != 3 or not all(math.isfinite(v) for v in fault.shift_m):
        raise ValueError("finite xyz metre shift required")
    if fault.kind is PositionKind.DROPOUT:
        if fault.count >= n:
            raise ValueError("cannot drop every position")
        return _rows(
            data, np.delete(np.arange(n), np.arange(fault.index, fault.index + fault.count))
        )
    if fault.kind is PositionKind.INTERPOLATED:
        knots = np.unique(np.r_[np.arange(0, n, fault.stride), n - 1])
        for name in data:
            if name != "time":
                data[name] = np.interp(data["time"], data["time"][knots], data[name][knots])
        return data
    _integer(fault.seed, "seed")
    if not math.isfinite(fault.noise_sd_m) or fault.noise_sd_m <= 0:
        raise ValueError("positive finite position noise SD required")
    shifts = (
        np.random.default_rng(fault.seed).normal(0, fault.noise_sd_m, (fault.count, 3))
        if fault.kind is PositionKind.NOISE
        else np.tile(fault.shift_m, (fault.count, 1))
    )
    rows = range(fault.index, fault.index + fault.count)
    if bundle.meta.device.device_class is DeviceClass.LPS:
        for i, axis in enumerate("xyz"):
            data[f"position_{axis}"][fault.index : fault.index + fault.count] += shifts[:, i]
    else:
        for j, i in enumerate(rows):
            frame = LocalFrame(
                float(data["latitude"][i]), float(data["longitude"][i]), float(data["height"][i])
            )
            fix = frame.to_geodetic(shifts[j])
            for name, value in zip(("latitude", "longitude", "height"), fix, strict=True):
                data[name][i] = value
    return data


def _inertial(
    bundle: SignalBundle, data: dict[str, FloatArray], fault: InertialFault
) -> dict[str, FloatArray]:
    if (
        not isinstance(fault.kind, InertialKind)
        or bundle.meta.device.device_class is not DeviceClass.IMU
    ):
        raise ValueError("inertial fault requires IMU and InertialKind")
    if fault.quantity not in {"specific_force", "angular_velocity"}:
        raise ValueError("unsupported inertial quantity")
    names = [f"{fault.quantity}_{a}" for a in "xyz"]
    if any(name not in data for name in names):
        raise ValueError("fault requires xyz inertial channels")
    if len(fault.bias) != 3 or not all(math.isfinite(v) for v in fault.bias):
        raise ValueError("finite xyz bias required")
    if not math.isfinite(fault.limit) or fault.limit <= 0:
        raise ValueError("positive finite saturation required")
    Frame(
        id="fault",
        kind="sensor",
        definition="prescribed misalignment",
        parent="original",
        rotation_to_parent=fault.rotation,
    )
    matrix = np.column_stack([data[name] for name in names])
    if fault.kind is InertialKind.BIAS:
        matrix += np.array(fault.bias)
    elif fault.kind is InertialKind.MISALIGNMENT:
        matrix = matrix @ np.array(fault.rotation).T
    else:
        matrix = np.clip(matrix, -fault.limit, fault.limit)
    for i, name in enumerate(names):
        data[name] = matrix[:, i].copy()
    return data


def _protocol(
    bundle: SignalBundle, data: dict[str, FloatArray], fault: ProtocolFault
) -> dict[str, FloatArray]:
    if not isinstance(fault.kind, ProtocolKind):
        raise ValueError("ProtocolKind required")
    if (
        not math.isfinite(fault.magnitude_n)
        or fault.magnitude_n <= 0
        or not math.isfinite(fault.window_s)
        or fault.window_s <= 0
    ):
        raise ValueError("positive finite protocol magnitude and window required")
    if fault.kind is ProtocolKind.SIDE_SWAP:
        left = [c for c in bundle.meta.channels if c.side is Side.LEFT]
        right = [c for c in bundle.meta.channels if c.side is Side.RIGHT]
        if not left or len(left) != len(right):
            raise ValueError("matched left/right measurements required")
        for left_channel, right_channel in zip(left, right, strict=True):
            if (left_channel.quantity, left_channel.axis) != (
                right_channel.quantity,
                right_channel.axis,
            ):
                raise ValueError("incompatible limb channel pair")
            data[left_channel.name], data[right_channel.name] = (
                data[right_channel.name].copy(),
                data[left_channel.name].copy(),
            )
        return data
    _index(fault.index, bundle.meta.n_rows)
    if bundle.meta.device.device_class is DeviceClass.CONTACT_MAT:
        if fault.kind is ProtocolKind.QUIET_MOTION:
            raise ValueError("quiet stance motion requires force observations")
        if fault.kind is ProtocolKind.STEP_OFF:
            if data["contact"][fault.index] != 1:
                raise ValueError("step-off must start from loaded contact")
            data["contact"][fault.index :] = 0
            return data
        i = fault.index
        if i + 1 >= bundle.meta.n_rows or data["contact"][i] != 0 or data["contact"][i + 1] != 1:
            raise ValueError("double contact requires an airborne interval")
        start, end = data["t_s"][i : i + 2]
        if fault.window_s >= end - start:
            raise ValueError("contact window must fit inside flight")
        mid = (start + end) / 2
        for name in list(data):
            new = (
                np.array([mid - fault.window_s / 2, mid + fault.window_s / 2])
                if name == "t_s"
                else np.array([1.0, 0.0])
            )
            data[name] = np.insert(data[name], i + 1, new)
        return data
    if bundle.meta.device.device_class is not DeviceClass.FORCE_PLATE:
        raise ValueError("protocol mechanics require force plate or contact observations")
    forces = [
        c.name for c in bundle.meta.channels if c.quantity is Quantity.FORCE and c.axis == "z"
    ]
    if not forces:
        raise ValueError("vertical forces required")
    times = data["time"]
    total = np.sum([data[name] for name in forces], axis=0)
    if fault.kind is ProtocolKind.STEP_OFF:
        if total[fault.index] <= 0:
            raise ValueError("step-off must start from loaded contact")
        for name in forces:
            data[name][fault.index :] = 0
        return data
    selected = (times >= times[fault.index]) & (times < times[fault.index] + fault.window_s)
    if fault.kind is ProtocolKind.DOUBLE_CONTACT:
        if np.any(np.abs(total[selected]) > 1e-9):
            raise ValueError("double contact requires an airborne interval")
        for name in forces:
            data[name][selected] = fault.magnitude_n / len(forces)
    else:
        if not np.allclose(total[selected], total[fault.index], rtol=0, atol=1e-9):
            raise ValueError("quiet motion must start in a flat quiet baseline")
        phase = (times[selected] - times[fault.index]) / fault.window_s
        for name in forces:
            data[name][selected] += fault.magnitude_n * np.sin(2 * np.pi * phase) / len(forces)
    return data


def inject(bundle: SignalBundle, fault: Fault) -> SignalBundle:
    """Return a new corrupted observation, preserving clean arrays and lineage."""
    if (
        bundle.meta.data_category is not DataCategory.SYNTHETIC
        or bundle.meta.vendor_reported
        or not any(
            tag in bundle.meta.provenance.transforms for tag in (OBSERVATION_STAGE, CORRUPTED_STAGE)
        )
    ):
        raise ValueError(
            "only RES-669 simulated/corrupted observations accepted; truth/vendor/empirical refused"
        )
    # Revalidate potentially mutable inputs at the trust boundary.
    meta = BundleMeta.model_validate(bundle.meta.model_dump())
    SignalBundle(meta, bundle.data)
    data = {name: values.copy() for name, values in bundle.data.items()}
    if isinstance(fault, ContinuousFault):
        data = _continuous(bundle, data, fault)
    elif isinstance(fault, EventFault):
        data = _event(bundle, data, fault)
    elif isinstance(fault, PositionFault):
        data = _position(bundle, data, fault)
    elif isinstance(fault, InertialFault):
        data = _inertial(bundle, data, fault)
    elif isinstance(fault, ProtocolFault):
        data = _protocol(bundle, data, fault)
    else:
        raise ValueError("unsupported fault type")
    if not data or not len(next(iter(data.values()))):
        raise ValueError("fault removed every sample")
    if all(np.array_equal(values, bundle[name]) for name, values in data.items()):
        raise ValueError("fault has no sample effect; unchanged data cannot be labelled corrupted")
    parameters = json.dumps(
        dict(type=type(fault).__name__, parameters=asdict(fault)), sort_keys=True, allow_nan=False
    )
    parent_hash = content_hash(bundle)
    meta.bundle_id = (
        "res669-corrupted-" + hashlib.sha256((parent_hash + parameters).encode()).hexdigest()[:24]
    )
    meta.data_sha256 = None
    meta.n_rows = len(next(iter(data.values())))
    meta.provenance.transforms = [
        tag for tag in meta.provenance.transforms if tag != OBSERVATION_STAGE
    ]
    if CORRUPTED_STAGE not in meta.provenance.transforms:
        meta.provenance.transforms.append(CORRUPTED_STAGE)
    meta.provenance.transforms.append("fault=" + parameters)
    meta.provenance.parents.append(Parent(bundle_id=bundle.meta.bundle_id, sha256=parent_hash))
    if meta.sampling is not None:
        times = data["time"]
        meta.sampling.t0_s = float(times[0])
        dt = np.diff(times)
        meta.sampling.measured_hz = (
            float(1 / np.median(dt)) if len(dt) > 0 and np.all(dt > 0) else None
        )
        meta.sampling.irregular = bool(
            len(dt) == 0
            or np.any(dt <= 0)
            or not np.allclose(dt, np.median(dt), rtol=1e-6, atol=1e-12)
        )
        if isinstance(fault, ContinuousFault) and fault.kind is ContinuousKind.RESAMPLE:
            meta.sampling.nominal_hz = meta.sampling.nominal_hz / fault.stride
    return SignalBundle(BundleMeta.model_validate(meta.model_dump()), data)
