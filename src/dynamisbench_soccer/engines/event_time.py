"""Time authority and interval arithmetic for native event streams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.signals.schema import BundleKind, DeviceClass, SignalBundle

Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]
Name = Annotated[str, Field(min_length=1)]


class EventRefusal(ValueError):
    """Event or timing authority is inadequate for a requested estimate."""


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class EdgeTiming(Policy):
    latency_s: Nonnegative
    latency_error_s: Nonnegative
    definition_error_s: Nonnegative = 0


class ClockTiming(Policy):
    resolution_s: Positive
    sync_latency_s: Nonnegative
    sync_latency_error_s: Nonnegative


class Timing(Policy):
    clock_model: Literal["shared", "wired", "wireless"]
    clocks: dict[Name, ClockTiming]
    authority: Name

    @model_validator(mode="after")
    def _authority(self) -> Timing:
        if self.authority.strip().lower() in {"", "unknown", "none", "unrecorded"}:
            raise ValueError("timing authority must identify a calibration or protocol")
        return self


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def __post_init__(self) -> None:
        if not all(map(math.isfinite, (self.value, self.low, self.high))):
            raise EventRefusal("non-finite timing or metric")

    def elapsed(self, start: Estimate, phase: str) -> Estimate:
        result = Estimate(self.value - start.value, self.low - start.high, self.high - start.low)
        if result.low <= 0:
            raise EventRefusal(f"{phase}: out of order or unresolved at declared timing precision")
        return result

    def shifted(self, offset: Estimate) -> Estimate:
        return Estimate(self.value + offset.value, self.low + offset.low, self.high + offset.high)


@dataclass(frozen=True)
class NativeEvent:
    bundle_id: str
    source_id: str
    row: int
    clock: str
    native_s: float


@dataclass(frozen=True)
class Event:
    time_s: Estimate
    native: tuple[NativeEvent, ...]


@dataclass(frozen=True)
class Discard:
    reason: str
    native: tuple[NativeEvent, ...]


class EventStream:
    """Read a source's native timestamps without reordering or resampling them."""

    def __init__(self, bundle: SignalBundle, device: DeviceClass, timing: Timing) -> None:
        meta = bundle.meta
        acquisition = meta.acquisition
        if meta.kind is not BundleKind.EVENTS or meta.device.device_class is not device:
            raise EventRefusal(f"expected source-native {device} events")
        if meta.data_category is DataCategory.ANALYTICAL:
            raise EventRefusal("analytical truth is a comparator, not instrument events")
        if acquisition is None:
            raise EventRefusal("missing source-native acquisition and clock authority")
        source = next((s for s in acquisition.sources if "t_s" in s.channels), None)
        if source is None or source.device.device_class is not device:
            raise EventRefusal("timestamp source does not match the event device")
        clocks = {c.id: c for c in acquisition.clocks}
        self.clock = clocks[source.clock]
        self.reference = clocks[acquisition.reference_clock]
        if any(
            c.source == "unknown" or c.epoch.strip().lower() in {"", "unknown"}
            for c in (self.clock, self.reference)
        ):
            raise EventRefusal("unknown clock source or epoch")
        self.alignment = next((a for a in acquisition.alignments if a.clock == source.clock), None)
        spec = timing.clocks.get(source.clock)
        if spec is None:
            raise EventRefusal(f"missing resolution and synchronization latency for {source.clock}")
        self.spec = spec
        if self.clock == self.reference:
            if self.spec.sync_latency_s or self.spec.sync_latency_error_s:
                raise EventRefusal("reference-clock events have no inter-clock sync latency")
        elif timing.clock_model == "shared" or self.alignment is None:
            raise EventRefusal("events require a common clock or an explicit synchronization")
        self.times = tuple(map(float, bundle["t_s"]))
        if any(b < a for a, b in pairwise(self.times)):
            raise EventRefusal(f"{meta.bundle_id}: out-of-order native timestamps")
        if self.alignment is not None:
            anchors = self.alignment.sync_events
            if (
                len(anchors) < 2
                or any(b <= a or y <= x for (a, x), (b, y) in pairwise(anchors))
                or self.times[0] < anchors[0][0]
                or self.times[-1] > anchors[-1][0]
            ):
                raise EventRefusal("synchronization requires ordered pulse pairs bracketing events")
        self.bundle_id = meta.bundle_id
        self.source = source

    def native(self, row: int) -> NativeEvent:
        return NativeEvent(self.bundle_id, self.source.id, row, self.clock.id, self.times[row])

    def event(self, row: int, edge: EdgeTiming) -> Event:
        native = self.times[row]
        seconds = native
        rate = 1.0
        sync_error = 0.0
        if (alignment := self.alignment) is not None:
            drift = alignment.drift_ppm * 1e-6
            rate += drift
            seconds += alignment.offset_s + (native - alignment.origin_s) * drift
            seconds -= self.spec.sync_latency_s
            sync_error = alignment.uncertainty_s + self.spec.sync_latency_error_s
        center = seconds - edge.latency_s
        error = rate * self.spec.resolution_s / 2 + sync_error
        error += edge.latency_error_s + edge.definition_error_s
        if error <= 0 or center - error == center or center + error == center:
            raise EventRefusal("timing bounds are below timestamp numerical precision")
        return Event(
            Estimate(center, center - error, center + error),
            (self.native(row),),
        )
