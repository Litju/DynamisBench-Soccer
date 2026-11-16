"""Calibration, unloaded zeroing, axes, clocks and alignment for force platforms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisbench_soccer.signals.schema import (
    BundleKind,
    BundleMeta,
    FloatArray,
    Quantity,
    Side,
    SignalBundle,
)

COMPONENTS = ("fx", "fy", "fz", "mx", "my", "mz")
Component = Literal["fx", "fy", "fz", "mx", "my", "mz"]


class PlateSpec(BaseModel):
    """Explicit matrix maps declared input units to N/N*m; SI inputs otherwise pass through.

    Metadata gains on already calibrated SI channels are historical provenance, not
    instructions to apply a raw calibration a second time. Zero offsets are in output
    component units, subtracted before rotation to the common field frame.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    plate: int = Field(ge=1)
    channels: tuple[str, ...] = ()
    input_units: tuple[str, ...] = ()
    output_components: tuple[Component, ...] = ()
    matrix: tuple[tuple[float, ...], ...] | None = None
    raw_offsets: tuple[float, ...] = ()
    zero_offsets: dict[Component, float] = Field(default_factory=dict)
    zero_window_s: tuple[float, float] | None = None
    rotation: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    origin_m: tuple[float, float, float] = (0, 0, 0)
    latency_s: float = Field(default=0, ge=0)
    saturation_limits: dict[Component, tuple[float, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_matrix(self) -> PlateSpec:
        r = np.array(self.rotation)
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(
            np.linalg.det(r), 1, atol=1e-12, rtol=0
        ):
            raise ValueError("plate rotation must be proper orthonormal")
        if self.matrix is not None:
            matrix = np.array(self.matrix)
            if (
                not self.channels
                or len(set(self.channels)) != len(self.channels)
                or not self.output_components
                or len(set(self.output_components)) != len(self.output_components)
            ):
                raise ValueError("matrix calibration needs unique ordered inputs and outputs")
            if len(self.input_units) != len(self.channels) or matrix.shape != (
                len(self.output_components),
                len(self.channels),
            ):
                raise ValueError("calibration dimensions and input units must match channel order")
            if (
                np.linalg.matrix_rank(matrix) < len(self.output_components)
                or np.linalg.cond(matrix) > 1e12
            ):
                raise ValueError("rank-deficient or ill-conditioned calibration matrix")
        elif self.input_units or self.output_components or self.raw_offsets:
            raise ValueError("matrix parameters require an explicit calibration matrix")
        if self.raw_offsets and len(self.raw_offsets) != len(self.channels):
            raise ValueError("raw offset count must match calibration inputs")
        if self.zero_window_s is not None and self.zero_window_s[1] <= self.zero_window_s[0]:
            raise ValueError("zero window must have positive duration")
        if self.zero_window_s is not None and self.zero_offsets:
            raise ValueError("choose an unloaded zero window or explicit offsets, not both")
        if any(lo >= hi for lo, hi in self.saturation_limits.values()):
            raise ValueError("saturation limits must increase")
        return self


@dataclass(frozen=True)
class PlateTrace:
    plate: int
    time_s: FloatArray
    wrench: FloatArray
    components: tuple[str, ...]
    moments_measured: bool
    side: Side
    offset: FloatArray
    offset_sd: FloatArray
    sync_uncertainty_s: float
    saturated: bool
    source: SignalBundle


@dataclass(frozen=True)
class Acquired:
    time_s: FloatArray
    plates: tuple[PlateTrace, ...]
    resampled: bool
    assumed_clock: bool

    @property
    def vertical_force_n(self) -> FloatArray:
        return np.sum([p.wrench[:, 2] for p in self.plates], axis=0)


def input_digest(bundle: SignalBundle) -> str:
    raw = bundle.meta.model_dump(mode="json")
    raw["data_sha256"] = None
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for values in bundle.data.values():
        digest.update(values.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def window_mask(time: FloatArray, window: tuple[float, float], minimum_s: float = 0) -> np.ndarray:
    start, end = window
    if not np.isfinite([start, end]).all() or end - start < minimum_s or end <= start:
        raise ValueError("invalid/insufficient window duration")
    dt = float(np.median(np.diff(time)))
    if start < time[0] - 1e-9 or end > time[-1] + dt + 1e-9:
        raise ValueError("window outside sampled interval")
    mask = (time >= start - 1e-10) & (time < end - 1e-10)
    if int(mask.sum()) < 2 or int(mask.sum()) * dt < minimum_s - 1e-9:
        raise ValueError("window contains insufficient samples")
    return mask


def _time(
    bundle: SignalBundle, channels: tuple[str, ...], latency_s: float
) -> tuple[FloatArray, float, str | None]:
    sampling = bundle.meta.sampling
    if sampling is None:
        raise ValueError("force platform requires sampling metadata")
    native = (
        bundle["time"].copy()
        if "time" in bundle.data
        else sampling.t0_s
        + np.arange(bundle.meta.n_rows) / (sampling.measured_hz or sampling.nominal_hz)
    )
    if len(native) < 3 or np.any(np.diff(native) <= 0):
        raise ValueError("at least three strictly increasing timestamps required")
    dt = np.diff(native)
    step = float(np.median(dt))
    if not np.allclose(dt, step, rtol=1e-6, atol=1e-12):
        raise ValueError("irregular/gapped plate timebase; explicit upstream resampling required")
    if abs(1 / step - sampling.nominal_hz) / sampling.nominal_hz > 0.01:
        raise ValueError("measured rate differs from nominal by more than 1 percent")
    acquisition = bundle.meta.acquisition
    if acquisition is None:
        return native - latency_s, 0, None
    sources = [s for s in acquisition.sources if set(channels) & set(s.channels)]
    clocks = {s.clock for s in sources}
    if len(clocks) != 1:
        raise ValueError("plate channels do not share one declared clock")
    clock = next(iter(clocks))
    reference = next(c for c in acquisition.clocks if c.id == acquisition.reference_clock)
    identity = reference.id + ":" + reference.epoch
    if clock == acquisition.reference_clock:
        return native - latency_s, 0, identity
    alignment = next((a for a in acquisition.alignments if a.clock == clock), None)
    if alignment is None:
        raise ValueError("plate clock lacks reference alignment")
    time = native + alignment.offset_s + (native - alignment.origin_s) * alignment.drift_ppm * 1e-6
    return time - latency_s, alignment.uncertainty_s, identity


def _plate(bundle: SignalBundle, spec: PlateSpec) -> tuple[PlateTrace, str | None]:
    if (
        bundle.meta.kind is not BundleKind.TIMESERIES
        or bundle.meta.device.device_class.value != "force_plate"
    ):
        raise ValueError("production engine accepts force-platform timeseries only")
    BundleMeta.model_validate(bundle.meta.model_dump())
    SignalBundle(bundle.meta, bundle.data)
    names = spec.channels or tuple(
        c.name
        for c in bundle.meta.channels
        if c.plate == spec.plate and c.quantity in {Quantity.FORCE, Quantity.TORQUE}
    )
    if not names:
        raise ValueError(f"plate {spec.plate}: no measurement channels")
    if len(names) != len(set(names)) or any(n not in bundle.data for n in names):
        raise ValueError("invalid/duplicate plate channels")
    specs = [bundle.meta.channel(n) for n in names]
    if any(c.plate != spec.plate for c in specs):
        raise ValueError("calibration input belongs to another plate")
    time, sync, identity = _time(bundle, names, spec.latency_s)
    raw = np.column_stack([bundle[n] for n in names])
    outputs: tuple[str, ...]
    if spec.matrix is not None:
        if tuple(c.unit for c in specs) != spec.input_units:
            raise ValueError("matrix input units differ from acquisition channels")
        calibrated = (raw - np.array(spec.raw_offsets or (0,) * len(names))) @ np.array(
            spec.matrix
        ).T
        outputs = spec.output_components
    else:
        outputs_list = []
        for c in specs:
            if c.axis is None or c.quantity not in {Quantity.FORCE, Quantity.TORQUE}:
                raise ValueError(
                    "SI passthrough needs force/torque axes; raw voltage needs a calibration matrix"
                )
            outputs_list.append(("f" if c.quantity is Quantity.FORCE else "m") + c.axis)
        outputs = tuple(outputs_list)
        if len(outputs) != len(set(outputs)):
            raise ValueError("duplicate plate component")
        calibrated = raw
    if "fz" not in outputs:
        raise ValueError("plate requires a vertical force component")
    if set(spec.zero_offsets) - set(outputs):
        raise ValueError("zero offset names an unavailable component")
    wrench = np.zeros((len(time), 6), dtype=np.float64)
    offset = np.zeros(6, dtype=np.float64)
    sd = offset.copy()
    for i, name in enumerate(outputs):
        j = COMPONENTS.index(name)
        wrench[:, j] = calibrated[:, i]
    for name, value in spec.zero_offsets.items():
        offset[COMPONENTS.index(name)] = value
    if spec.zero_window_s is not None:
        mask = window_mask(time, spec.zero_window_s)
        for name in outputs:
            j = COMPONENTS.index(name)
            offset[j] = float(np.mean(wrench[mask, j]))
            sd[j] = float(np.std(wrench[mask, j], ddof=1))
    if set(spec.saturation_limits) - set(outputs):
        raise ValueError("saturation limit names an unavailable component")
    saturated = any(
        np.any((wrench[:, COMPONENTS.index(c)] <= lo) | (wrench[:, COMPONENTS.index(c)] >= hi))
        for c, (lo, hi) in spec.saturation_limits.items()
    )
    wrench -= offset
    rotation = np.array(spec.rotation)
    force_complete = set(("fx", "fy", "fz")) <= set(outputs)
    moment_complete = set(("mx", "my", "mz")) <= set(outputs)
    if not np.allclose(rotation, np.eye(3), atol=1e-12, rtol=0) and not force_complete:
        raise ValueError("rotating a partial force vector is unsupported")
    if any(name.startswith("m") for name in outputs) and not moment_complete:
        raise ValueError("moment reconstruction requires all three moment components")
    wrench[:, :3] = wrench[:, :3] @ rotation.T
    if moment_complete:
        wrench[:, 3:] = wrench[:, 3:] @ rotation.T
    # Transport measured moments to the common field origin; missing moments stay missing.
    if moment_complete:
        wrench[:, 3:] += np.cross(np.array(spec.origin_m), wrench[:, :3])
    sides = {c.side for c in specs if c.quantity is Quantity.FORCE and c.axis == "z"}
    side = next(iter(sides)) if len(sides) == 1 else Side.NONE
    trace = PlateTrace(
        spec.plate,
        time,
        wrench,
        tuple(outputs),
        moment_complete,
        side,
        offset,
        sd,
        sync,
        saturated,
        bundle,
    )
    return trace, identity


def acquire(
    bundles: SignalBundle | tuple[SignalBundle, ...],
    plates: tuple[PlateSpec, ...] = (),
    *,
    alignment: Literal["strict", "linear_overlap"] = "strict",
    assume_shared_clock: bool = False,
) -> Acquired:
    inputs = (bundles,) if isinstance(bundles, SignalBundle) else bundles
    if not inputs or len(inputs) > 2:
        raise ValueError("one or two acquisition bundles required")
    if alignment not in {"strict", "linear_overlap"}:
        raise ValueError("unsupported alignment method")
    ids = sorted(
        {
            c.plate
            for b in inputs
            for c in b.meta.channels
            if c.plate is not None and c.quantity in {Quantity.FORCE, Quantity.VOLTAGE}
        }
    )
    if not 1 <= len(ids) <= 2:
        raise ValueError("only single and dual force platforms supported")
    selected = (
        tuple(PlateSpec.model_validate(p.model_dump()) for p in plates)
        if plates
        else tuple(PlateSpec(plate=p) for p in ids)
    )
    if len({p.plate for p in selected}) != len(selected) or {p.plate for p in selected} != set(ids):
        raise ValueError("specify each acquired plate exactly once")
    traces = []
    identities = []
    for spec in selected:
        candidates = [b for b in inputs if any(c.plate == spec.plate for c in b.meta.channels)]
        if len(candidates) != 1:
            raise ValueError("plate IDs must be unique across acquisition bundles")
        trace, identity = _plate(candidates[0], spec)
        traces.append(trace)
        identities.append(identity)
    assumed = len(traces) == 2 and any(i is None for i in identities)
    if assumed and not assume_shared_clock:
        raise ValueError(
            "dual plates need declared synchronization or explicit shared-clock assumption"
        )
    declared = {i for i in identities if i is not None}
    if len(declared) > 1:
        raise ValueError("plate reference clocks/epochs disagree")
    grid = traces[0].time_s.copy()
    resampled = False
    if len(traces) == 2:
        other = traces[1].time_s
        same = len(grid) == len(other) and np.allclose(grid, other, rtol=0, atol=1e-9)
        if not same:
            if alignment == "strict":
                raise ValueError(
                    "dual plate grids differ; request explicit linear_overlap alignment"
                )
            lo = max(t.time_s[0] for t in traces)
            hi = min(t.time_s[-1] for t in traces)
            grid = grid[(grid >= lo) & (grid <= hi)]
            if len(grid) < 3:
                raise ValueError("insufficient synchronized plate overlap")
            resampled = True
    aligned = []
    for p in traces:
        values = (
            p.wrench.copy()
            if len(grid) == len(p.time_s) and np.allclose(grid, p.time_s, atol=1e-9, rtol=0)
            else np.column_stack([np.interp(grid, p.time_s, p.wrench[:, j]) for j in range(6)])
        )
        aligned.append(
            PlateTrace(
                p.plate,
                grid.copy(),
                values,
                p.components,
                p.moments_measured,
                p.side,
                p.offset.copy(),
                p.offset_sd.copy(),
                p.sync_uncertainty_s,
                p.saturated,
                p.source,
            )
        )
    return Acquired(grid, tuple(aligned), resampled, assumed)


def cop(
    acquired: Acquired, threshold_n: float = 20
) -> tuple[FloatArray | None, dict[int, FloatArray | None]]:
    if not np.isfinite(threshold_n) or threshold_n <= 0:
        raise ValueError("positive finite COP load threshold required")

    def from_wrench(wrench: FloatArray) -> FloatArray:
        output = np.full((len(wrench), 2), np.nan, dtype=np.float64)
        valid = wrench[:, 2] > threshold_n
        output[valid, 0] = -wrench[valid, 4] / wrench[valid, 2]
        output[valid, 1] = wrench[valid, 3] / wrench[valid, 2]
        return output

    per_plate = {
        p.plate: from_wrench(p.wrench) if p.moments_measured else None for p in acquired.plates
    }
    combined = (
        from_wrench(np.sum([p.wrench for p in acquired.plates], axis=0))
        if all(p.moments_measured for p in acquired.plates)
        else None
    )
    return combined, per_plate
