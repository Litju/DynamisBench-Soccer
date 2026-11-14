"""Canonical signal bundle, schema version 1.

A *bundle* is one recorded trial (or one event list, or one session summary). It is
stored as two files with a common stem:

* ``<stem>.parquet`` - the samples (columns = channels), float64, SI units
* ``<stem>.json``    - :class:`BundleMeta`, the metadata sidecar, including the
  SHA-256 of the Parquet file

Three kinds exist:

* ``timeseries`` - uniformly sampled channels (force plates, dynamometers, radar/laser,
  LPT). Time is implicit: ``t_i = t0_s + i / nominal_hz``. An optional measured
  ``time`` channel may be stored when the device provides timestamps.
* ``events``     - timestamped events (timing gates). Required columns: ``t_s`` and
  ``distance_m``.
* ``summary``    - one row per session or drill of pre-aggregated metrics (GNSS/IMU
  summaries). These are context inputs, never raw signals.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.registry.models import V01_TESTS

SCHEMA_VERSION: Literal[1] = 1
FloatArray = npt.NDArray[np.float64]

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class BundleKind(StrEnum):
    TIMESERIES = "timeseries"
    EVENTS = "events"
    SUMMARY = "summary"


class DeviceClass(StrEnum):
    FORCE_PLATE = "force_plate"
    NORDIC_DYNAMOMETER = "nordic_dynamometer"
    HIP_DYNAMOMETER = "hip_dynamometer"
    TIMING_GATES = "timing_gates"
    RADAR_LASER = "radar_laser"
    LPT = "lpt"
    GNSS_IMU = "gnss_imu"
    MANUAL_PROTOCOL = "manual_protocol"


class Quantity(StrEnum):
    FORCE = "force"
    TORQUE = "torque"
    VELOCITY = "velocity"
    DISTANCE = "distance"
    DISPLACEMENT = "displacement"
    TIME = "time"
    POSITION = "position"  # e.g. centre of pressure
    ANGLE = "angle"
    COUNT = "count"
    DIMENSIONLESS = "dimensionless"


#: Canonical unit per quantity. Adapters convert into these; nothing else is stored.
CANONICAL_UNIT: dict[Quantity, str] = {
    Quantity.FORCE: "N",
    Quantity.TORQUE: "N*m",
    Quantity.VELOCITY: "m/s",
    Quantity.DISTANCE: "m",
    Quantity.DISPLACEMENT: "m",
    Quantity.TIME: "s",
    Quantity.POSITION: "m",
    Quantity.ANGLE: "deg",
    Quantity.COUNT: "1",
    Quantity.DIMENSIONLESS: "1",
}


class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    NONE = "none"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChannelSpec(_Strict):
    name: str = Field(description="column name, snake_case, unique within the bundle")
    quantity: Quantity
    unit: str
    side: Side = Side.NONE
    plate: int | None = Field(default=None, ge=1, description="force plate number")
    axis: Literal["x", "y", "z"] | None = None
    description: str = ""

    @field_validator("name")
    @classmethod
    def _snake(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError(f"channel name must be snake_case: {value!r}")
        return value

    @model_validator(mode="after")
    def _canonical_unit(self) -> ChannelSpec:
        expected = CANONICAL_UNIT[self.quantity]
        if self.unit != expected:
            raise ValueError(
                f"channel {self.name}: unit {self.unit!r} is not canonical {expected!r} "
                f"for {self.quantity}; adapters must convert"
            )
        return self


class Subject(_Strict):
    pseudonym: str = Field(pattern=r"^[A-Za-z0-9_-]{3,64}$")
    body_mass_kg: float | None = Field(default=None, gt=20, lt=200)
    sex: Literal["male", "female", "not_recorded"] = "not_recorded"
    birth_year: int | None = Field(default=None, ge=1950, le=2030)


class Session(_Strict):
    session_id: str = Field(min_length=1)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    match_day: str | None = Field(default=None, pattern=r"^MD([+-][0-9])?$")
    trial_index: int = Field(default=1, ge=1)


class Device(_Strict):
    device_class: DeviceClass
    manufacturer: str | None = None
    model: str | None = None
    firmware: str | None = None
    serial_hash: str | None = Field(default=None, description="hash, never the raw serial")


class Sampling(_Strict):
    nominal_hz: float = Field(gt=0)
    measured_hz: float | None = Field(default=None, gt=0)
    t0_s: float = 0.0


class Calibration(_Strict):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    method: str
    notes: str = ""


class Zeroing(_Strict):
    performed: bool
    method: str = ""
    offsets: dict[str, float] = Field(default_factory=dict, description="per channel, N")


class Geometry(_Strict):
    plate_layout: Literal["single", "dual_side_by_side", "dual_in_line"] | None = None
    summed_force_policy: Literal["sum_fz_all_plates"] | None = None
    lever_arm_m: float | None = Field(default=None, gt=0, lt=1.0)
    knee_angle_deg: float | None = None
    hip_flexion_deg: float | None = None
    gate_height_m: float | None = Field(default=None, gt=0, lt=2.0)


class Protocol(_Strict):
    protocol_id: str = Field(min_length=1)
    arm_swing: Literal["hands_on_hips", "free_arms", "not_applicable"] = "not_applicable"
    start_mode: Literal["standing", "crouched", "flying", "not_applicable"] = "not_applicable"
    action: str | None = Field(default=None, description="e.g. adduction, abduction")
    drop_height_m: float | None = Field(default=None, gt=0, lt=1.5)


class Parent(_Strict):
    bundle_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Provenance(_Strict):
    created_by: str = Field(description="tool@version, e.g. adapter:generic_forceplate_csv@0.1")
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parents: list[Parent] = Field(default_factory=list)
    transforms: list[str] = Field(default_factory=list, description="ordered, human-readable")


class VendorValue(_Strict):
    value: float
    unit: str
    definition: str = "unknown"


class BundleMeta(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    bundle_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    kind: BundleKind
    test_id: str
    data_category: DataCategory
    subject: Subject
    session: Session
    device: Device
    protocol: Protocol
    sampling: Sampling | None = None
    channels: list[ChannelSpec] = Field(min_length=1)
    n_rows: int = Field(ge=1)
    calibration: list[Calibration] = Field(default_factory=list)
    zeroing: Zeroing | None = None
    geometry: Geometry = Field(default_factory=Geometry)
    provenance: Provenance
    vendor_reported: dict[str, VendorValue] = Field(
        default_factory=dict,
        description="vendor-computed metrics, kept apart; never used as reference (DB-07)",
    )
    data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("test_id")
    @classmethod
    def _known_test(cls, value: str) -> str:
        if value not in V01_TESTS:
            raise ValueError(f"unknown test_id {value!r}")
        return value

    @model_validator(mode="after")
    def _kind_rules(self) -> BundleMeta:
        names = [c.name for c in self.channels]
        if len(names) != len(set(names)):
            raise ValueError("duplicate channel names")
        if self.kind is BundleKind.TIMESERIES and self.sampling is None:
            raise ValueError("timeseries bundles require sampling")
        if self.kind is BundleKind.EVENTS:
            if self.sampling is not None:
                raise ValueError("events bundles must not declare sampling")
            if not {"t_s", "distance_m"} <= set(names):
                raise ValueError("events bundles require t_s and distance_m channels")
        if self.device.device_class is DeviceClass.FORCE_PLATE:
            plates = {c.plate for c in self.channels if c.quantity is Quantity.FORCE}
            if None in plates:
                raise ValueError("force-plate force channels must declare their plate")
            if len(plates) > 1 and self.geometry.summed_force_policy is None:
                raise ValueError("multi-plate bundles must declare summed_force_policy")
        return self

    def channel(self, name: str) -> ChannelSpec:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)


class SignalBundle:
    """Metadata plus float64 column arrays of equal length."""

    def __init__(self, meta: BundleMeta, data: dict[str, FloatArray]) -> None:
        names = [c.name for c in meta.channels]
        if list(data) != names:
            raise ValueError(f"data columns {list(data)} must match channels {names} in order")
        lengths = {len(v) for v in data.values()}
        if lengths != {meta.n_rows}:
            raise ValueError(f"column lengths {sorted(lengths)} != n_rows {meta.n_rows}")
        for name, values in data.items():
            if values.dtype != np.float64 or values.ndim != 1:
                raise ValueError(f"{name}: columns must be 1-D float64")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name}: non-finite values are not allowed")
        self.meta = meta
        self.data = data

    def __getitem__(self, name: str) -> FloatArray:
        return self.data[name]

    def time(self) -> FloatArray:
        """Sample times for timeseries bundles (measured ``time`` channel if present)."""
        if "time" in self.data:
            return self.data["time"]
        if self.meta.sampling is None:
            raise ValueError("bundle has no sampling")
        n = self.meta.n_rows
        return (
            self.meta.sampling.t0_s + np.arange(n, dtype=np.float64) / self.meta.sampling.nominal_hz
        )

    def channels_where(self, quantity: Quantity, side: Side | None = None) -> list[ChannelSpec]:
        return [
            c
            for c in self.meta.channels
            if c.quantity is quantity and (side is None or c.side is side)
        ]

    def total_vertical_force(self) -> FloatArray:
        """Sum of all vertical force channels (``summed_force_policy``)."""
        fz = [c.name for c in self.meta.channels if c.quantity is Quantity.FORCE and c.axis == "z"]
        if not fz:
            raise ValueError("no vertical force channels")
        total = np.zeros(self.meta.n_rows, dtype=np.float64)
        for name in fz:
            total += self.data[name]
        return total
