"""Affine device clocks; timestamps and measurement latency are separate quantities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dynamisbench_soccer.signals.schema import Alignment, FloatArray


@dataclass(frozen=True)
class DeviceClock:
    offset_s: float = 0
    drift_ppm: float = 0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.offset_s)
            or not math.isfinite(self.drift_ppm)
            or self.drift_ppm <= -1e6
        ):
            raise ValueError("finite offset and direction-preserving clock drift required")

    def reference_time(self, device_s: FloatArray) -> FloatArray:
        return device_s * (1 + self.drift_ppm * 1e-6) + self.offset_s

    def device_time(self, reference_s: FloatArray) -> FloatArray:
        return (reference_s - self.offset_s) / (1 + self.drift_ppm * 1e-6)

    def alignment(self, end_s: float) -> Alignment:
        if not math.isfinite(end_s) or end_s <= 0:
            raise ValueError("positive finite synchronization span required")
        scale = 1 + self.drift_ppm * 1e-6
        return Alignment(
            clock="device",
            reference_clock="reference",
            method="prescribed affine clock; two exact sync pulses",
            offset_s=self.offset_s,
            drift_ppm=self.drift_ppm,
            uncertainty_s=1e-12,
            sync_events=[(-self.offset_s / scale, 0), ((end_s - self.offset_s) / scale, end_s)],
        )
