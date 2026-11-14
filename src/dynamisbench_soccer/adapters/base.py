"""Adapter interface, capability manifest and shared metadata handling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from dynamisbench_soccer.adapters.csvformat import CsvFormatError, ParsedCsv, parse
from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    BundleMeta,
    ChannelSpec,
    Device,
    DeviceClass,
    FloatArray,
    Geometry,
    Protocol,
    Provenance,
    Sampling,
    Session,
    SignalBundle,
    Subject,
    VendorValue,
    Zeroing,
)

#: Metadata keys every adapter understands. Unknown keys are rejected (typo guard).
COMMON_KEYS = {
    "dbs-generic-csv",
    "test_id",
    "data_category",
    "bundle_id",
    "subject",
    "body_mass_kg",
    "sex",
    "birth_year",
    "session_id",
    "date",
    "match_day",
    "trial",
    "protocol_id",
    "manufacturer",
    "model",
    "firmware",
    "calibration_date",
    "calibration_method",
    "zeroing",
    "zeroing_method",
    "sampling_rate_hz",
}
SAMPLING_TOLERANCE = 0.01  # relative; larger deviations of measured vs nominal are recorded


class AdapterError(ValueError):
    """An export could not be converted to a valid canonical bundle."""


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    device_classes: tuple[DeviceClass, ...]
    test_ids: tuple[str, ...]
    kind: BundleKind
    required_columns: str
    optional_metadata: tuple[str, ...] = ()
    notes: str = ""


class Adapter(ABC):
    capabilities: ClassVar[Capabilities]
    extra_keys: ClassVar[frozenset[str]] = frozenset()

    def read(self, path: Path) -> SignalBundle:
        try:
            parsed = parse(path)
            self._check_keys(parsed)
            test_id = self._require(parsed, "test_id")
            if test_id not in self.capabilities.test_ids:
                raise AdapterError(
                    f"{self.capabilities.name} does not support test_id {test_id!r} "
                    f"(supports {', '.join(self.capabilities.test_ids)})"
                )
            return self.convert(parsed, test_id)
        except AdapterError:
            raise
        except CsvFormatError as exc:
            raise AdapterError(f"{path.name}: {exc}") from exc
        except ValidationError as exc:
            raise AdapterError(f"{path.name}: invalid metadata: {exc}") from exc
        except ValueError as exc:  # e.g. bad enum value or integer in metadata
            raise AdapterError(f"{path.name}: {exc}") from exc

    @abstractmethod
    def convert(self, parsed: ParsedCsv, test_id: str) -> SignalBundle: ...

    # --- helpers ------------------------------------------------------------------------
    def _check_keys(self, parsed: ParsedCsv) -> None:
        unknown = set(parsed.meta) - COMMON_KEYS - self.extra_keys
        if unknown:
            raise AdapterError(f"unknown metadata keys: {sorted(unknown)}")

    @staticmethod
    def _require(parsed: ParsedCsv, key: str) -> str:
        value = parsed.meta.get(key)
        if not value:
            raise AdapterError(f"missing required metadata '# {key}: ...'")
        return value

    @staticmethod
    def _float(parsed: ParsedCsv, key: str) -> float | None:
        value = parsed.meta.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise AdapterError(f"metadata {key} must be numeric, got {value!r}") from exc

    def sampling(self, parsed: ParsedCsv, transforms: list[str]) -> Sampling:
        """Resolve sampling from a ``time`` column and/or ``sampling_rate_hz``."""
        declared = self._float(parsed, "sampling_rate_hz")
        time = parsed.column("time")
        if time is None:
            if declared is None:
                raise AdapterError("missing sampling rate: no time column and no sampling_rate_hz")
            return Sampling(nominal_hz=declared)
        dt = np.diff(time.values)
        if len(dt) == 0 or np.any(dt <= 0):
            raise AdapterError("time column must be strictly increasing")
        measured = float(1.0 / np.median(dt))
        if declared is None:
            transforms.append(f"nominal_hz inferred from time column ({measured:.6g} Hz)")
            return Sampling(
                nominal_hz=round(measured, 6), measured_hz=measured, t0_s=float(time.values[0])
            )
        if abs(measured - declared) / declared > SAMPLING_TOLERANCE:
            transforms.append(f"measured {measured:.6g} Hz differs from declared {declared:g} Hz")
        return Sampling(nominal_hz=declared, measured_hz=measured, t0_s=float(time.values[0]))

    def build(
        self,
        parsed: ParsedCsv,
        test_id: str,
        kind: BundleKind,
        device_class: DeviceClass,
        channels: list[ChannelSpec],
        data: dict[str, FloatArray],
        transforms: list[str],
        *,
        sampling: Sampling | None = None,
        geometry: Geometry | None = None,
        protocol_extra: dict[str, object] | None = None,
    ) -> SignalBundle:
        m = parsed.meta
        subject = self._require(parsed, "subject")
        session_id = self._require(parsed, "session_id")
        trial = int(m.get("trial", "1"))
        bundle_id = m.get("bundle_id") or f"{test_id}-{subject}-{session_id}-t{trial}"
        try:
            category = DataCategory(self._require(parsed, "data_category"))
        except ValueError as exc:
            raise AdapterError(f"invalid data_category {m.get('data_category')!r}") from exc
        calibration = []
        if m.get("calibration_date"):
            calibration.append(
                {
                    "date": m["calibration_date"],
                    "method": m.get("calibration_method", "not recorded"),
                }
            )
        zeroing = None
        if "zeroing" in m:
            if m["zeroing"] not in {"yes", "no"}:
                raise AdapterError("zeroing must be 'yes' or 'no'")
            zeroing = Zeroing(performed=m["zeroing"] == "yes", method=m.get("zeroing_method", ""))
        vendor = {
            k: VendorValue(value=v, unit=u, definition="vendor-reported; definition unknown")
            for k, (v, u) in parsed.vendor.items()
        }
        meta = BundleMeta(
            bundle_id=bundle_id,
            kind=kind,
            test_id=test_id,
            data_category=category,
            subject=Subject(
                pseudonym=subject,
                body_mass_kg=self._float(parsed, "body_mass_kg"),
                sex=m.get("sex", "not_recorded"),
                birth_year=int(m["birth_year"]) if "birth_year" in m else None,
            ),
            session=Session(
                session_id=session_id,
                date=self._require(parsed, "date"),
                match_day=m.get("match_day"),
                trial_index=trial,
            ),
            device=Device(
                device_class=device_class,
                manufacturer=m.get("manufacturer"),
                model=m.get("model"),
                firmware=m.get("firmware"),
            ),
            protocol=Protocol.model_validate(
                {"protocol_id": self._require(parsed, "protocol_id"), **(protocol_extra or {})}
            ),
            sampling=sampling,
            channels=channels,
            n_rows=parsed.n_rows,
            calibration=calibration,
            zeroing=zeroing,
            geometry=geometry or Geometry(),
            provenance=Provenance(
                created_by=f"adapter:{self.capabilities.name}@{self.capabilities.version}",
                source_sha256=parsed.source_sha256,
                transforms=[*parsed.conversions, *transforms],
            ),
            vendor_reported=vendor,
        )
        return SignalBundle(meta, data)


ADAPTERS: dict[str, Adapter] = {}


def register(adapter: Adapter) -> Adapter:
    ADAPTERS[adapter.capabilities.name] = adapter
    return adapter


def get(name: str) -> Adapter:
    from dynamisbench_soccer.adapters import generic  # noqa: F401  (registers adapters)

    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise AdapterError(f"unknown adapter {name!r}; known: {sorted(ADAPTERS)}") from exc


def all_adapters() -> dict[str, Adapter]:
    from dynamisbench_soccer.adapters import generic  # noqa: F401

    return dict(ADAPTERS)
