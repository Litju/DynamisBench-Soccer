"""Parser for the documented *DBS generic CSV* export format (docs/acquisition-boundary.md).

Layout::

    # dbs-generic-csv: 1
    # key: value            <- metadata lines (any order)
    # vendor.<metric>[unit]: value   <- vendor-computed values (kept apart)
    col_a[unit],col_b[unit]  <- header: snake_case name + unit in brackets
    1.0,2.0
    ...

Every column needs a unit. Values must be numeric. Rows must be complete.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dynamisbench_soccer.signals.schema import FloatArray, Quantity

MAGIC = "dbs-generic-csv"
SUPPORTED_FORMAT_VERSIONS = {"1"}
_COLUMN = re.compile(r"^([a-z][a-z0-9_]*)\[([^\]]+)\]$")
_VENDOR = re.compile(r"^vendor\.([a-z][a-z0-9_]*)\[([^\]]+)\]$")

#: unit -> (quantity family, factor to canonical unit). Distance-family units map to
#: DISTANCE; adapters re-label them as DISPLACEMENT or POSITION where appropriate.
UNITS: dict[str, tuple[Quantity, float]] = {
    "N": (Quantity.FORCE, 1.0),
    "kN": (Quantity.FORCE, 1000.0),
    "kgf": (Quantity.FORCE, 9.80665),
    "lbf": (Quantity.FORCE, 4.4482216152605),
    "N*m": (Quantity.TORQUE, 1.0),
    "Nm": (Quantity.TORQUE, 1.0),
    "m": (Quantity.DISTANCE, 1.0),
    "cm": (Quantity.DISTANCE, 0.01),
    "mm": (Quantity.DISTANCE, 0.001),
    "km": (Quantity.DISTANCE, 1000.0),
    "s": (Quantity.TIME, 1.0),
    "ms": (Quantity.TIME, 0.001),
    "min": (Quantity.TIME, 60.0),
    "m/s": (Quantity.VELOCITY, 1.0),
    "km/h": (Quantity.VELOCITY, 1 / 3.6),
    "deg": (Quantity.ANGLE, 1.0),
    "1": (Quantity.COUNT, 1.0),
}


class CsvFormatError(ValueError):
    """The file does not conform to the DBS generic CSV format."""


@dataclass
class Column:
    name: str
    unit: str
    quantity: Quantity
    factor: float
    values: FloatArray  # already converted to the canonical unit


@dataclass
class ParsedCsv:
    meta: dict[str, str]
    columns: list[Column]
    vendor: dict[str, tuple[float, str]] = field(default_factory=dict)
    source_sha256: str = ""
    conversions: list[str] = field(default_factory=list)

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def n_rows(self) -> int:
        return len(self.columns[0].values) if self.columns else 0


def parse(path: Path) -> ParsedCsv:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CsvFormatError("file is not UTF-8") from exc
    lines = text.splitlines()
    meta: dict[str, str] = {}
    vendor: dict[str, tuple[float, str]] = {}
    body_start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            body_start = i
            break
        key, sep, value = line[1:].partition(":")
        if not sep:
            raise CsvFormatError(f"line {i + 1}: metadata must be '# key: value'")
        key, value = key.strip(), value.strip()
        vm = _VENDOR.match(key)
        if vm:
            try:
                vendor[vm.group(1)] = (float(value), vm.group(2))
            except ValueError as exc:
                raise CsvFormatError(f"line {i + 1}: vendor value not numeric") from exc
            continue
        if key in meta:
            raise CsvFormatError(f"line {i + 1}: duplicate metadata key {key!r}")
        meta[key] = value
    else:
        raise CsvFormatError("no header row")

    if meta.get(MAGIC) not in SUPPORTED_FORMAT_VERSIONS:
        raise CsvFormatError(f"missing or unsupported '# {MAGIC}: <version>' line")

    reader = csv.reader(io.StringIO("\n".join(lines[body_start:])))
    header = next(reader, None)
    if not header:
        raise CsvFormatError("no header row")
    specs = []
    for col in (h.strip() for h in header):
        m = _COLUMN.match(col)
        if not m:
            raise CsvFormatError(f"column {col!r} must look like name[unit]")
        name, unit = m.groups()
        if unit not in UNITS:
            raise CsvFormatError(f"column {name}: unknown unit {unit!r}")
        specs.append((name, unit))
    names = [n for n, _ in specs]
    if len(names) != len(set(names)):
        raise CsvFormatError("duplicate column names")

    rows: list[list[float]] = []
    for line_no, row in enumerate(reader, start=body_start + 2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(specs):
            raise CsvFormatError(f"line {line_no}: expected {len(specs)} fields, got {len(row)}")
        try:
            values = [float(cell) for cell in row]
        except ValueError as exc:
            raise CsvFormatError(f"line {line_no}: non-numeric value") from exc
        if not all(math.isfinite(v) for v in values):
            raise CsvFormatError(f"line {line_no}: non-finite value")
        rows.append(values)
    if not rows:
        raise CsvFormatError("no data rows")

    matrix = np.asarray(rows, dtype=np.float64)
    columns, conversions = [], []
    for j, (name, unit) in enumerate(specs):
        quantity, factor = UNITS[unit]
        columns.append(Column(name, unit, quantity, factor, matrix[:, j] * factor))
        if factor != 1.0:
            conversions.append(f"{name}: {unit} -> canonical (x{factor:g})")
    return ParsedCsv(
        meta=meta,
        columns=columns,
        vendor=vendor,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        conversions=conversions,
    )
