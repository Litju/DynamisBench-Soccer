"""Source-version-specific adapter: ``shorts`` ``data-raw/laser-gun-data.csv`` (RES-720).

What the source *is* (``shorts`` ``R/data-laser-gun.R`` at the pinned commit): one 35 m
standing-start sprint by a youth basketball player, recorded by a Noptel CMP3 laser
at 2.56 kHz. Musclelab v10.232.107.5298 fitted a polynomial distance-time model and
resampled it to 1 kHz. Every CSV column is therefore **manufacturer-processed**; the
file holds no native optical range samples.

The package's ``data-raw/laser-gun-data.R`` then derives a *different* object
(velocity offset removal, trimming, dv/dt and Butterworth accelerations). Those are
**package-derived** and are documented here, not reproduced or ingested.

This module accepts only the exact pinned bytes (or bytes an explicit test adapter
names) and refuses anything else rather than guessing.
"""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Protocol as _Protocol

import numpy as np
from pydantic import ValidationError

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.external.models import (
    FileRole,
    SignalStage,
    SourceLock,
    date_mentioned,
    date_problem,
)
from dynamisbench_soccer.external.rights import (
    RightsBlocked,
    RightsCheck,
    require_canonical_admission,
)
from dynamisbench_soccer.signals.schema import (
    CANONICAL_UNIT,
    Acquisition,
    AcquisitionSource,
    BundleKind,
    BundleMeta,
    ChannelSpec,
    Clock,
    Device,
    DeviceClass,
    FloatArray,
    Geometry,
    Protocol,
    Provenance,
    Quantity,
    Sampling,
    Session,
    SignalBundle,
    Subject,
)

SOURCE_ID = "shorts-laser"
ADAPTER_ID = "shorts_laser_csv"
ADAPTER_VERSION = "1.0.0"
DATA_PATH = "data-raw/laser-gun-data.csv"
#: The exact source version this adapter was written and verified against.
PINNED_REF = "5f8ce5554b5aca0e857199e0eb50bf66d7089130"
PINNED_SHA256 = "52b48d928460a656775e8df2aa1824c84b214040e281d4992754fbac73f3c6ba"

EXPECTED_HEADER = ("time", "distance", "velocity", "raw_velocity")
MAX_BYTES = 1024 * 1024
MAX_ROWS = 100_000
#: Rate of the vendor-resampled grid as published; native acquisition was 2.56 kHz.
PUBLISHED_RESAMPLED_HZ = 1000.0
NATIVE_ACQUISITION_HZ = 2560.0
MUSCLELAB_VERSION = "Musclelab v10.232.107.5298 (Ergotest Technology AS)"
DEVICE_MANUFACTURER = "Noptel Oy"
DEVICE_MODEL = "CMP3 Distance Sensor"

_NUMBER = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


class SourceIntegrityError(RuntimeError):
    """Bytes are not the pinned source version."""


class SourceFormatError(ValueError):
    """Bytes are the expected file type but violate its strict format contract."""


@dataclass(frozen=True)
class ColumnSemantics:
    source_column: str
    channel: str
    quantity: Quantity
    stage: SignalStage
    meaning: str

    @property
    def description(self) -> str:
        return f"[{self.stage}] source column '{self.source_column}': {self.meaning}"


SOURCE_COLUMNS: tuple[ColumnSemantics, ...] = (
    ColumnSemantics(
        "time",
        "time",
        Quantity.TIME,
        SignalStage.MANUFACTURER_PROCESSED,
        "Musclelab 1 kHz resampled, trial-relative time grid; vendor-constructed, "
        "not the native 2.56 kHz sensor clock",
    ),
    ColumnSemantics(
        "distance",
        "distance",
        Quantity.DISTANCE,
        SignalStage.MANUFACTURER_PROCESSED,
        "Musclelab polynomial distance-time model evaluated on the 1 kHz grid; "
        "not native optical range samples",
    ),
    ColumnSemantics(
        "velocity",
        "velocity",
        Quantity.VELOCITY,
        SignalStage.MANUFACTURER_PROCESSED,
        "Musclelab smoothed (step-averaged) velocity",
    ),
    ColumnSemantics(
        "raw_velocity",
        "velocity_vendor_unsmoothed",
        Quantity.VELOCITY,
        SignalStage.MANUFACTURER_PROCESSED,
        "Musclelab unsmoothed velocity export labelled 'raw' by the source; vendor "
        "derivation undocumented; NOT native range data",
    ),
)

#: Package transforms in ``data-raw/laser-gun-data.R``; documented, never applied here.
PACKAGE_DERIVED_NOT_INGESTED: tuple[tuple[str, str], ...] = (
    ("velocity_offset", "velocity <- velocity - min(velocity) after trimming time > 0.375 s"),
    ("trim", "rows with time <= 0.375 s, then time <= 0.414 s, removed (4805 rows remain)"),
    ("raw_acceleration", "dv/dt of smoothed velocity: c(NA, diff(velocity) / diff(time))"),
    ("butter_acceleration", "4th-order 1 Hz low-pass Butterworth, filtfilt, of raw_acceleration"),
)
PACKAGE_DOCUMENTED_ROWS = 4805
PACKAGE_TRIM_S = 0.414


@dataclass(frozen=True)
class SourceTable:
    sha256: str
    size_bytes: int
    columns: dict[str, FloatArray]  # keyed by *source* column name, in source order

    @property
    def n_rows(self) -> int:
        return len(self.columns["time"])


def _read(path: Path) -> bytes:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SourceIntegrityError(f"{path}: must be a regular file")
    if info.st_size > MAX_BYTES:
        raise SourceFormatError(f"{path}: {info.st_size} B exceeds {MAX_BYTES} B limit")
    return path.read_bytes()


def parse(path: Path, expected_sha256: str = PINNED_SHA256) -> SourceTable:
    """Hash-check, then strictly parse. No coercion, imputation or column guessing."""
    data = _read(path)
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected_sha256:
        raise SourceIntegrityError(
            f"{path.name}: sha256 {sha[:12]}... is not the pinned {expected_sha256[:12]}..."
        )
    if data.startswith(b"\xef\xbb\xbf"):
        raise SourceFormatError("byte-order mark not allowed")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SourceFormatError(f"non-ASCII byte at offset {exc.start}") from exc
    newline = "\r\n" if "\r\n" in text else "\n"
    if "\r" in text.replace("\r\n", ""):
        raise SourceFormatError("stray carriage return")
    lines = text.split(newline)
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or tuple(lines[0].split(",")) != EXPECTED_HEADER:
        raise SourceFormatError(f"header must be exactly {','.join(EXPECTED_HEADER)}")
    rows = lines[1:]
    if len(rows) < 2 or len(rows) > MAX_ROWS:
        raise SourceFormatError(f"row count {len(rows)} outside [2, {MAX_ROWS}]")
    values = np.empty((len(rows), len(EXPECTED_HEADER)), dtype=np.float64)
    for i, line in enumerate(rows, start=2):
        fields = line.split(",")
        if len(fields) != len(EXPECTED_HEADER):
            raise SourceFormatError(f"line {i}: {len(fields)} fields, expected 4")
        for j, field in enumerate(fields):
            if not _NUMBER.match(field):
                raise SourceFormatError(f"line {i}: {EXPECTED_HEADER[j]} {field!r} is not finite")
            values[i - 2, j] = float(field)
    if not np.all(np.isfinite(values)):
        raise SourceFormatError("non-finite value after parsing")
    columns = {name: np.ascontiguousarray(values[:, j]) for j, name in enumerate(EXPECTED_HEADER)}
    return SourceTable(sha256=sha, size_bytes=len(data), columns=columns)


def profile(table: SourceTable) -> dict[str, float | int | bool]:
    """Gate-2 shape audit: grain, clock, ordering, quantisation and value ranges."""
    t = table.columns["time"]
    dt = np.diff(t)
    d = table.columns["distance"]
    dd = np.diff(d)
    positive_steps = dd[dd > 0]
    out: dict[str, float | int | bool] = {
        "n_rows": table.n_rows,
        "n_columns": len(table.columns),
        "size_bytes": table.size_bytes,
        "t_first_s": float(t[0]),
        "t_last_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "dt_min_s": float(dt.min()),
        "dt_median_s": float(np.median(dt)),
        "dt_max_s": float(dt.max()),
        "n_nonpositive_dt": int(np.sum(dt <= 0)),
        "n_duplicate_time": int(table.n_rows - np.unique(t).size),
        "uniform_grid": bool(np.allclose(dt, np.median(dt), rtol=1e-6, atol=1e-12)),
        "grid_hz_from_median_dt": float(1.0 / np.median(dt)),
        "n_distance_decreases": int(np.sum(dd < 0)),
        "distance_resolution_m": float(positive_steps.min()) if positive_steps.size else 0.0,
        "missing_or_nonfinite": 0,  # parse() refuses any; recorded for the audit trail
    }
    for name, values in table.columns.items():
        out[f"{name}_min"] = float(values.min())
        out[f"{name}_max"] = float(values.max())
    return out


# --- schema-v2 conversion ----------------------------------------------------------------
class SessionEvidenceError(ValueError):
    """A claimed acquisition date is not supported by its pinned evidence bytes."""


@dataclass(frozen=True)
class BoundSessionDate:
    """A session date proven against a hash-verified ``acquisition_evidence`` file."""

    date: str
    evidence_path: str
    evidence_sha256: str
    excerpt: str
    reviewed_by: str


def bind_session_date(lock: SourceLock, files: Mapping[str, Path]) -> BoundSessionDate | None:
    """Return the lock's date claim only if its evidence bytes prove it, else raise.

    ``None`` means no claim exists. Re-hashes the evidence file (even though acquisition
    already verified it), decodes it strictly and requires the excerpt verbatim.
    """
    claim = lock.acquisition_date
    if claim is None:
        return None
    entry = lock.file(claim.evidence_path)
    if entry.role is not FileRole.ACQUISITION_EVIDENCE:
        raise SessionEvidenceError(f"{entry.path}: not an acquisition_evidence file")
    path = files.get(entry.path)
    if path is None:
        raise SessionEvidenceError(f"{entry.path}: evidence file was not acquired")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.sha256 or len(data) != entry.size_bytes:
        raise SessionEvidenceError(f"{entry.path}: bytes differ from the pinned evidence")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SessionEvidenceError(f"{entry.path}: evidence is not UTF-8 text") from exc
    if _date.fromisoformat(claim.reviewed_on) > _date.today():
        raise SessionEvidenceError("evidence review date is in the future")
    if text.count(claim.excerpt) != 1:
        raise SessionEvidenceError(f"{entry.path}: excerpt must occur verbatim exactly once")
    if not date_mentioned(claim.excerpt, claim.date):
        raise SessionEvidenceError("excerpt does not state the claimed date")
    # Judge the excerpt in context: the whole line(s) it sits on must pass the same check,
    # so an excerpt cut out of a negating or multi-date sentence cannot bind.
    at = text.index(claim.excerpt)
    start = text.rfind("\n", 0, at) + 1
    end = text.find("\n", at + len(claim.excerpt))
    context = text[start : end if end != -1 else len(text)]
    problem = date_problem(context, claim.date)
    if problem is not None:
        raise SessionEvidenceError(f"{entry.path}: excerpt context fails: {problem}")
    return BoundSessionDate(claim.date, entry.path, entry.sha256, claim.excerpt, claim.reviewed_by)


class TimeBaseStatement(_Protocol):
    """The part of ``timebase.TimeBaseAgreement`` that conversion records."""

    def statement(self) -> str: ...


@dataclass(frozen=True)
class ConversionOutcome:
    bundle: SignalBundle | None
    blockers: list[str]
    determinations: list[tuple[str, str]]

    @property
    def converted(self) -> bool:
        return self.bundle is not None


#: The test identity is a measurement-protocol mapping, never a soccer task admission.
SPRINT_FV_MAPPING = (
    "PROVISIONAL kinematic protocol mapping only, not soccer task admission: a maximal 35 m "
    "standing-start linear sprint with continuous laser distance matches the sprint_fv "
    "acquisition protocol. Only kinematic estimands (vmax, tau) are supportable; force/power "
    "estimands (f0, pmax, fv_slope, rf_max, drf) are unsupported (no body mass). Population: "
    "one youth basketball player, no soccer claim (see protocol_id). Any task use requires "
    "separate review; data usage stays cross-validation-only."
)

#: Recorded with every conversion attempt, converted or not.
DETERMINATIONS: tuple[tuple[str, str], ...] = (
    ("test_id=sprint_fv", SPRINT_FV_MAPPING),
    (
        "session_date",
        "accepted only from a source-lock acquisition_date claim: a pinned "
        "acquisition_evidence file (SHA-256), a verbatim excerpt stating the date and a named "
        "reviewer; never from a CLI flag or free text",
    ),
    (
        "conversion_authority",
        "an empirical v2 bundle is a canonical artifact; it is built only after canonical "
        "admission is verified in rights.json",
    ),
    (
        "device_class=laser",
        f"{DEVICE_MANUFACTURER} {DEVICE_MODEL}; native {NATIVE_ACQUISITION_HZ:g} Hz",
    ),
    ("geometry", "mounting_height_m and aiming_point unpublished: declared unknown, not inferred"),
    ("calibration", "no calibration record published: calibration_unknown_reason, no certificate"),
    ("clock", "vendor trial-relative grid; source 'unknown', absolute epoch unpublished"),
    ("sampling", "nominal 1000 Hz is the published resampled rate; measured from the time column"),
    ("subject", "anonymous pseudonym; sex, body mass and birth year not recorded"),
    ("signal_stage", "all channels manufacturer_processed; no measured_native channel exists"),
)


def bundle_id(pinned_ref: str = PINNED_REF) -> str:
    return f"shorts-laser-{pinned_ref[:12]}-laser-gun-data"


def convert(
    table: SourceTable,
    lock: SourceLock,
    evidence_files: Mapping[str, Path],
    *,
    admission: RightsCheck,
    time_base: TimeBaseStatement,
) -> ConversionOutcome:
    """Build a strict schema-v2 bundle, or report every blocker; never fill gaps.

    Two independent authorities are required: a :class:`RightsCheck` issued by
    ``rights.verify`` that permits canonical admission, and a session date bound to pinned
    evidence in the source lock. Neither is accepted as a plain value.
    """
    pinned_ref = lock.pinned_ref
    blockers: list[str] = []
    try:
        require_canonical_admission(admission, lock)
    except RightsBlocked:
        blockers.append(
            "canonical admission: rights.json does not verify canonical admission; an empirical "
            "schema-v2 bundle is a canonical artifact and is not built for quarantined data"
        )
    session: BoundSessionDate | None = None
    try:
        session = bind_session_date(lock, evidence_files)
    except SessionEvidenceError as exc:
        blockers.append(f"session.date: acquisition-date evidence rejected: {exc}")
    else:
        if session is None:
            blockers.append(
                "session.date: no source-bound, reviewed acquisition-date evidence is pinned "
                "in the source lock (the date is not published by the source); schema-v2 "
                "Session.date is mandatory and no date may be invented (RES-719 falsifier matrix)"
            )
    t = table.columns["time"]
    dt = np.diff(t)
    if np.any(dt <= 0):
        blockers.append("time: source timestamps are not strictly increasing")
    determinations = [*DETERMINATIONS, ("temporal_alignment", time_base.statement())]
    if blockers or session is None:
        return ConversionOutcome(None, blockers, determinations)

    names = [c.channel for c in SOURCE_COLUMNS]
    channels = [
        ChannelSpec(
            name=c.channel,
            quantity=c.quantity,
            unit=CANONICAL_UNIT[c.quantity],
            description=c.description,
        )
        for c in SOURCE_COLUMNS
    ]
    device = Device(
        device_class=DeviceClass.LASER, manufacturer=DEVICE_MANUFACTURER, model=DEVICE_MODEL
    )
    geometry = Geometry(unknown=["mounting_height_m", "aiming_point"])
    uniform = bool(np.allclose(dt, np.median(dt), rtol=1e-6, atol=1e-12))
    raw_format = (
        f"{MUSCLELAB_VERSION} export: polynomial distance-time model resampled from "
        f"{NATIVE_ACQUISITION_HZ:g} Hz to {PUBLISHED_RESAMPLED_HZ:g} Hz, committed as shorts "
        f"{DATA_PATH} @ {pinned_ref[:12]}"
    )
    try:
        meta = BundleMeta(
            bundle_id=bundle_id(pinned_ref),
            kind=BundleKind.TIMESERIES,
            test_id="sprint_fv",
            data_category=DataCategory.EMPIRICAL,
            subject=Subject(pseudonym="shorts-laser-athlete-01"),
            session=Session(session_id="shorts-laser-gun-data", date=session.date),
            device=device,
            protocol=Protocol(
                protocol_id="shorts-laser-35m-standing-start-youth-basketball-non-soccer",
                start_mode="standing",
            ),
            sampling=Sampling(
                nominal_hz=PUBLISHED_RESAMPLED_HZ,
                measured_hz=float(1.0 / np.median(dt)),
                t0_s=float(t[0]),
                irregular=not uniform,
            ),
            channels=channels,
            n_rows=table.n_rows,
            geometry=geometry,
            provenance=Provenance(
                created_by=f"adapter:{ADAPTER_ID}@{ADAPTER_VERSION}",
                source_sha256=table.sha256,
                transforms=[
                    f"[measured_native, unavailable] {DEVICE_MODEL} acquisition at "
                    f"{NATIVE_ACQUISITION_HZ:g} Hz (per shorts documentation)",
                    f"[manufacturer_processed] {MUSCLELAB_VERSION}: polynomial distance-time "
                    "model, 1 kHz resampling, smoothed and unsmoothed velocity export",
                    f"shorts @ {pinned_ref[:12]} {DATA_PATH} used unchanged",
                    f"session date {session.date} from pinned {session.evidence_path} "
                    f"(sha256 {session.evidence_sha256}), excerpt {session.excerpt!r}, "
                    f"reviewed by {session.reviewed_by}",
                    f"test_id sprint_fv: {SPRINT_FV_MAPPING}",
                    f"temporal alignment: {time_base.statement()}",
                    f"adapter {ADAPTER_ID}@{ADAPTER_VERSION}: column raw_velocity renamed "
                    "velocity_vendor_unsmoothed; no resampling, filtering, offset removal, "
                    "trimming or imputation",
                    "NOT applied (package_derived): "
                    + "; ".join(f"{k}: {v}" for k, v in PACKAGE_DERIVED_NOT_INGESTED),
                ],
            ),
            acquisition=Acquisition(
                clocks=[
                    Clock(
                        id="musclelab_trial",
                        source="unknown",
                        epoch="Musclelab trial-relative time; absolute epoch and native "
                        "2.56 kHz sensor clock unpublished",
                    )
                ],
                reference_clock="musclelab_trial",
                sources=[
                    AcquisitionSource(
                        id="laser",
                        device=device,
                        clock="musclelab_trial",
                        channels=names,
                        geometry=geometry,
                        calibration_unknown_reason="no calibration record is published by "
                        "the source; none is assumed",
                        raw_format=raw_format,
                        source_sha256=table.sha256,
                    )
                ],
            ),
            acquisition_unknown_reason=None,
        )
        data = {c.channel: table.columns[c.source_column].copy() for c in SOURCE_COLUMNS}
        bundle = SignalBundle(meta, data)
    except (ValidationError, ValueError) as exc:
        return ConversionOutcome(None, [f"schema-v2 validation: {exc}"], determinations)
    return ConversionOutcome(bundle, [], determinations)
