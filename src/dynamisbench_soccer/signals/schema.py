"""Canonical signal bundle, schema version 2.

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
from datetime import date
from enum import StrEnum
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.registry.models import V01_TESTS

SCHEMA_VERSION: Literal[2] = 2
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
    GNSS_IMU = "gnss_imu"  # legacy summaries only
    GNSS = "gnss"
    LPS = "lps"
    IMU = "imu"
    ENCODER = "encoder"
    LOAD_CELL = "load_cell"
    ISOMETRIC_DYNAMOMETER = "isometric_dynamometer"
    CONTACT_MAT = "contact_mat"
    RADAR = "radar"
    LASER = "laser"
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
    ACCELERATION = "acceleration"
    SPECIFIC_FORCE = "specific_force"
    ANGULAR_VELOCITY = "angular_velocity"
    MAGNETIC_FLUX_DENSITY = "magnetic_flux_density"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    VOLTAGE = "voltage"


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
    Quantity.ACCELERATION: "m/s^2",
    Quantity.SPECIFIC_FORCE: "m/s^2",
    Quantity.ANGULAR_VELOCITY: "rad/s",
    Quantity.MAGNETIC_FLUX_DENSITY: "T",
    Quantity.LATITUDE: "deg",
    Quantity.LONGITUDE: "deg",
    Quantity.VOLTAGE: "V",
}


class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    NONE = "none"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ChannelSpec(_Strict):
    name: str = Field(description="column name, snake_case, unique within the bundle")
    quantity: Quantity
    unit: str
    side: Side = Side.NONE
    plate: int | None = Field(default=None, ge=1, description="force plate number")
    axis: Literal["x", "y", "z"] | None = None
    description: str = ""
    frame: str | None = None

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
    irregular: bool = False


class Calibration(_Strict):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    method: str
    notes: str = ""
    channel: str | None = None
    input_unit: str | None = None
    output_unit: str | None = None
    gain: float | None = None
    offset: float | None = None
    bias: tuple[float, float, float] | None = None
    scale: tuple[float, float, float] | None = None
    cosine_angle_rad: float | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    reference: str | None = None

    @field_validator("date")
    @classmethod
    def _date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


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

    attachment_point: str | None = None
    cable_anchors_m: list[tuple[float, float, float]] = Field(default_factory=list)
    cable_angle_rad: float | None = None
    encoder_counts_per_m: float | None = Field(default=None, gt=0)
    rig_description: str | None = None
    rig_compliance_m_per_n: float | None = Field(default=None, ge=0)
    beam_type: Literal["single", "dual", "multiple", "unknown"] | None = None
    trigger_method: str | None = None
    mounting_height_m: float | None = Field(default=None, ge=0)
    aiming_point: str | None = None
    lps_anchors_m: dict[str, tuple[float, float, float]] = Field(default_factory=dict)
    geodetic_datum: Literal["WGS84"] | None = None
    sensor_placement: str | None = None
    unknown: list[str] = Field(default_factory=list, description="missing geometry, never inferred")


class Frame(_Strict):
    id: str = Field(min_length=1)
    kind: Literal["sensor", "segment", "field", "cable", "geodetic"]
    definition: str = Field(min_length=1)
    parent: str | None = None
    rotation_to_parent: (
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | None
    ) = None

    @model_validator(mode="after")
    def _rotation(self) -> Frame:
        if self.rotation_to_parent is not None:
            r = np.array(self.rotation_to_parent)
            if (
                self.parent is None
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-10, rtol=0)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-10, rtol=0)
            ):
                raise ValueError("rotation must be proper orthonormal with a parent frame")
        return self


class Clock(_Strict):
    id: str = Field(min_length=1)
    source: Literal["device", "host", "gnss", "unknown"]
    epoch: str = Field(min_length=1, description="time origin; unknown must be explicit")
    drift_ppm: float | None = None


class Alignment(_Strict):
    clock: str
    reference_clock: str
    method: str = Field(min_length=1)
    offset_s: float
    drift_ppm: float = 0
    origin_s: float = 0
    uncertainty_s: float = Field(ge=0)
    sync_events: list[tuple[float, float]] = Field(
        min_length=1,
        description="paired source/reference timestamps; ref=t+offset+(t-origin)*drift*1e-6",
    )

    @model_validator(mode="after")
    def _events(self) -> Alignment:
        if self.drift_ppm <= -1e6:
            raise ValueError("clock alignment must preserve time direction")
        for source, reference in self.sync_events:
            predicted = source + self.offset_s + (source - self.origin_s) * self.drift_ppm * 1e-6
            if abs(reference - predicted) > self.uncertainty_s + 1e-12:
                raise ValueError("sync event disagrees with declared clock alignment")
        return self


class AcquisitionSource(_Strict):
    id: str = Field(min_length=1)
    device: Device
    clock: str
    channels: list[str] = Field(min_length=1)
    geometry: Geometry
    calibration: list[Calibration] = Field(default_factory=list)
    calibration_unknown_reason: str | None = None
    raw_format: str = Field(min_length=1)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _calibration(self) -> AcquisitionSource:
        required_geometry = {
            DeviceClass.FORCE_PLATE: ("plate_layout",),
            DeviceClass.LPT: ("attachment_point", "cable_anchors_m"),
            DeviceClass.ENCODER: ("encoder_counts_per_m", "attachment_point"),
            DeviceClass.LOAD_CELL: ("rig_description", "lever_arm_m"),
            DeviceClass.ISOMETRIC_DYNAMOMETER: ("rig_description", "lever_arm_m"),
            DeviceClass.NORDIC_DYNAMOMETER: ("lever_arm_m", "knee_angle_deg"),
            DeviceClass.HIP_DYNAMOMETER: ("lever_arm_m", "hip_flexion_deg"),
            DeviceClass.TIMING_GATES: ("gate_height_m", "beam_type", "trigger_method"),
            DeviceClass.CONTACT_MAT: ("trigger_method",),
            DeviceClass.RADAR: ("mounting_height_m", "aiming_point"),
            DeviceClass.LASER: ("mounting_height_m", "aiming_point"),
            DeviceClass.RADAR_LASER: ("mounting_height_m", "aiming_point"),
            DeviceClass.GNSS: ("geodetic_datum",),
            DeviceClass.LPS: ("lps_anchors_m",),
            DeviceClass.IMU: ("sensor_placement",),
        }
        for field in required_geometry.get(self.device.device_class, ()):
            value = getattr(self.geometry, field)
            if (value is None or value == [] or value == {}) and field not in self.geometry.unknown:
                raise ValueError(
                    f"{self.device.device_class} geometry needs {field} or explicit unknown"
                )
        if not self.calibration and not self.calibration_unknown_reason:
            raise ValueError("calibration records or explicit missing-calibration reason required")
        return self


class Acquisition(_Strict):
    clocks: list[Clock] = Field(min_length=1)
    reference_clock: str
    sources: list[AcquisitionSource] = Field(min_length=1)
    frames: list[Frame] = Field(default_factory=list)
    alignments: list[Alignment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _links(self) -> Acquisition:
        ids = [c.id for c in self.clocks]
        if len(ids) != len(set(ids)) or self.reference_clock not in ids:
            raise ValueError("clock IDs must be unique and include reference_clock")
        source_ids = [s.id for s in self.sources]
        if len(source_ids) != len(set(source_ids)) or any(s.clock not in ids for s in self.sources):
            raise ValueError("source IDs must be unique with known clocks")
        aligned = [a.clock for a in self.alignments]
        if len(aligned) != len(set(aligned)):
            raise ValueError("duplicate clock alignments")
        for a in self.alignments:
            if (
                a.clock not in ids
                or a.clock == self.reference_clock
                or a.reference_clock != self.reference_clock
            ):
                raise ValueError("alignments must map known clocks directly to reference_clock")
        if set(ids) - {self.reference_clock} != set(aligned):
            raise ValueError("every non-reference clock requires alignment")
        frames = {f.id: f for f in self.frames}
        if len(frames) != len(self.frames):
            raise ValueError("duplicate frames")
        for f in self.frames:
            seen = {f.id}
            parent = f.parent
            while parent is not None:
                if parent not in frames or parent in seen:
                    raise ValueError("unknown or cyclic parent frame")
                seen.add(parent)
                parent = frames[parent].parent
        return self


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
    schema_version: Literal[2] = SCHEMA_VERSION
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
    acquisition: Acquisition | None = None
    acquisition_unknown_reason: str | None = (
        "v1-compatible producer: acquisition details unrecorded"
    )
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
            required = (
                {"t_s", "contact"}
                if self.device.device_class is DeviceClass.CONTACT_MAT
                else {"t_s", "distance_m"}
            )
            if not required <= set(names):
                raise ValueError(
                    f"events bundles require {' and '.join(sorted(required))} channels"
                )
        if self.device.device_class is DeviceClass.FORCE_PLATE:
            plates = {c.plate for c in self.channels if c.quantity is Quantity.FORCE}
            if None in plates:
                raise ValueError("force-plate force channels must declare their plate")
            if len(plates) > 1 and self.geometry.summed_force_policy is None:
                raise ValueError("multi-plate bundles must declare summed_force_policy")
        if self.kind is BundleKind.SUMMARY and self.sampling is not None:
            raise ValueError("summary bundles must not declare sampling")
        if self.device.device_class is DeviceClass.GNSS_IMU and self.kind is not BundleKind.SUMMARY:
            raise ValueError("legacy gnss_imu is summary-only; use gnss, lps or imu for raw data")
        if self.sampling is not None and self.sampling.irregular and "time" not in names:
            raise ValueError("irregular sampling requires measured time channel")
        for c in self.channels:
            if c.name in {"time", "t_s"} and c.quantity is not Quantity.TIME:
                raise ValueError("timestamp columns must have time quantity and seconds units")
        if self.acquisition is None:
            if not self.acquisition_unknown_reason:
                raise ValueError("missing acquisition needs an explicit reason")
            if self.device.device_class in {
                DeviceClass.GNSS,
                DeviceClass.LPS,
                DeviceClass.IMU,
                DeviceClass.ENCODER,
                DeviceClass.LOAD_CELL,
                DeviceClass.ISOMETRIC_DYNAMOMETER,
                DeviceClass.CONTACT_MAT,
                DeviceClass.RADAR,
                DeviceClass.LASER,
            }:
                raise ValueError("new raw device classes require acquisition contracts")
        else:
            assigned = [n for source in self.acquisition.sources for n in source.channels]
            if len(assigned) != len(set(assigned)) or set(assigned) != set(names):
                raise ValueError("acquisition sources must partition bundle channels")
            frames = {f.id for f in self.acquisition.frames}
            if any(c.frame is not None and c.frame not in frames for c in self.channels):
                raise ValueError("unknown channel frame")
            for source in self.acquisition.sources:
                channels = [self.channel(n) for n in source.channels]
                quantities = {c.quantity for c in channels}
                cls = source.device.device_class
                required_quantities = {
                    DeviceClass.GNSS: {Quantity.LATITUDE, Quantity.LONGITUDE, Quantity.POSITION},
                    DeviceClass.LPS: {Quantity.POSITION},
                    DeviceClass.IMU: {Quantity.SPECIFIC_FORCE, Quantity.ANGULAR_VELOCITY},
                    DeviceClass.ENCODER: {Quantity.COUNT},
                    DeviceClass.RADAR: {Quantity.VELOCITY},
                    DeviceClass.LASER: {Quantity.DISTANCE},
                    DeviceClass.CONTACT_MAT: {Quantity.TIME, Quantity.DIMENSIONLESS},
                }
                if not required_quantities.get(cls, set()) <= quantities:
                    raise ValueError(f"{cls}: missing raw quantities")
                if cls in {
                    DeviceClass.LOAD_CELL,
                    DeviceClass.ISOMETRIC_DYNAMOMETER,
                    DeviceClass.FORCE_PLATE,
                    DeviceClass.NORDIC_DYNAMOMETER,
                    DeviceClass.HIP_DYNAMOMETER,
                } and not quantities & {Quantity.FORCE, Quantity.VOLTAGE}:
                    raise ValueError(f"{cls}: needs force or raw voltage")
                if cls is DeviceClass.LPT and not quantities & {
                    Quantity.DISTANCE,
                    Quantity.DISPLACEMENT,
                }:
                    raise ValueError("LPT needs cable distance or displacement")
                for quantity in (
                    {Quantity.SPECIFIC_FORCE, Quantity.ANGULAR_VELOCITY}
                    if cls is DeviceClass.IMU
                    else {Quantity.POSITION}
                    if cls is DeviceClass.LPS
                    else set()
                ):
                    components = [c for c in channels if c.quantity is quantity]
                    if {c.axis for c in components} != {"x", "y", "z"} or any(
                        c.frame is None for c in components
                    ):
                        raise ValueError(f"{cls}: requires framed xyz {quantity}")
                for record in source.calibration:
                    if record.channel is not None and record.channel not in source.channels:
                        raise ValueError("calibration channel not owned by source")
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
        for c in meta.channels:
            values = data[c.name]
            if c.quantity is Quantity.LATITUDE and np.any(np.abs(values) > 90):
                raise ValueError("latitude outside [-90, 90]")
            if c.quantity is Quantity.LONGITUDE and np.any(np.abs(values) > 180):
                raise ValueError("longitude outside [-180, 180]")
            if c.quantity is Quantity.COUNT and np.any(values != np.floor(values)):
                raise ValueError("raw counts must be integers")
        if (
            meta.device.device_class is DeviceClass.CONTACT_MAT
            and "contact" in data
            and not np.all(np.isin(data["contact"], [0, 1]))
        ):
            raise ValueError("contact state must be binary")
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
