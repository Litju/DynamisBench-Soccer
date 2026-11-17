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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dynamisbench_soccer.external.models import SignalStage
from dynamisbench_soccer.signals.schema import FloatArray, Quantity

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
