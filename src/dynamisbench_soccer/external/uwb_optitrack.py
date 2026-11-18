"""Source-version adapter: Zenodo 3452007 UWB/OptiTrack tables (RES-721).

Reads ``UWB_<Scenario>_dataset.csv`` members of the pinned archive, from in-memory
bytes. The BLE/UWB tables are pinned but have no reference system, so they are not
parsed here.

Column semantics come from the depositor README. Nothing undeclared is filled in:

* ``CIR1-4``, ``PSA1-4`` and ``d1-4``: device-reported per-anchor channel features and
  ranges (manufacturer-processed; anchor positions are not published).
* ``UWB_x``, ``UWB_y``: the depositor's UWB localisation estimate.
* ``OPT_x``, ``OPT_y``: the OptiTrack reference system output, reduced to 2D by the
  depositor. It is a separate measurement, not ground truth.
* ``SessionID``: integer session index 1-5. **There is no timestamp column.**
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from dynamisbench_soccer.external.models import SignalStage
from dynamisbench_soccer.external.planar_profile import PlanarProfile, PlanarSession, PlanarSystem
from dynamisbench_soccer.signals.schema import FloatArray

SOURCE_ID = "uwb-optitrack"
ADAPTER_ID = "uwb_optitrack_csv"
ADAPTER_VERSION = "1.0.0"
HEADER = (
    "SessionID",
    "CIR1",
    "CIR2",
    "CIR3",
    "CIR4",
    "PSA1",
    "PSA2",
    "PSA3",
    "PSA4",
    "d1",
    "d2",
    "d3",
    "d4",
    "UWB_x",
    "UWB_y",
    "OPT_x",
    "OPT_y",
)
SCENARIO_SUFFIX = {"walk": "UWB_Walk_dataset.csv", "trolley": "UWB_Trolley_dataset.csv"}
#: Sample counts stated in README_data.md, which (as verified) count the header line too.
README_COUNTS = {"walk": 2797, "trolley": 3202}
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 200_000
_NUMBER = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")

COLUMN_STAGES: dict[str, tuple[SignalStage, str]] = {
    "CIR": (SignalStage.MANUFACTURER_PROCESSED, "device-reported channel impulse response"),
    "PSA": (SignalStage.MANUFACTURER_PROCESSED, "device-reported preamble symbol accumulation"),
    "d": (SignalStage.MANUFACTURER_PROCESSED, "device-reported tag-anchor range"),
    "UWB": (SignalStage.PACKAGE_DERIVED, "depositor's UWB localisation estimate"),
    "OPT": (SignalStage.MANUFACTURER_PROCESSED, "OptiTrack system output reduced to 2D"),
}


class UwbFormatError(ValueError):
    """Bytes violate the strict format contract of the UWB/OptiTrack tables."""


@dataclass(frozen=True)
class UwbTable:
    scenario: str
    member: str
    member_sha256: str
    session: np.ndarray  # int64
    device: FloatArray  # (n, 2) UWB_x, UWB_y
    reference: FloatArray  # (n, 2) OPT_x, OPT_y
    ranges: FloatArray  # (n, 4) d1-d4

    @property
    def n_rows(self) -> int:
        return len(self.session)


def member_for(scenario: str, names: list[str]) -> str:
    matches = [n for n in names if n.rsplit("/", 1)[-1] == SCENARIO_SUFFIX[scenario]]
    if len(matches) != 1:
        raise UwbFormatError(f"{scenario}: expected one {SCENARIO_SUFFIX[scenario]} member")
    return matches[0]


def parse(scenario: str, member: str, member_sha256: str, data: bytes) -> UwbTable:
    """Strict: exact header, 17 numeric fields per row, integer non-decreasing sessions."""
    if len(data) > MAX_BYTES:
        raise UwbFormatError(f"{member}: exceeds {MAX_BYTES} bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise UwbFormatError("byte-order mark not allowed")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise UwbFormatError(f"non-ASCII byte at offset {exc.start}") from exc
    if "\r" in text:
        raise UwbFormatError("carriage returns not allowed")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or tuple(lines[0].split(",")) != HEADER:
        raise UwbFormatError(f"header must be exactly {','.join(HEADER)}")
    rows = lines[1:]
    if not 1 <= len(rows) <= MAX_ROWS:
        raise UwbFormatError(f"row count {len(rows)} outside [1, {MAX_ROWS}]")
    values = np.empty((len(rows), len(HEADER)), dtype=np.float64)
    for i, line in enumerate(rows, start=2):
        fields = line.split(",")
        if len(fields) != len(HEADER):
            raise UwbFormatError(f"line {i}: {len(fields)} fields, expected {len(HEADER)}")
        for j, f in enumerate(fields):
            if not _NUMBER.match(f):
                raise UwbFormatError(f"line {i}: {HEADER[j]} {f!r} is not a finite number")
            values[i - 2, j] = float(f)
    session = values[:, 0]
    if np.any(session != np.round(session)) or np.any(session < 1):
        raise UwbFormatError("SessionID must be a positive integer")
    if np.any(np.diff(session) < 0):
        raise UwbFormatError("SessionID must be non-decreasing (rows grouped by session)")
    return UwbTable(
        scenario=scenario,
        member=member,
        member_sha256=member_sha256,
        session=session.astype(np.int64),
        device=np.ascontiguousarray(values[:, 13:15]),
        reference=np.ascontiguousarray(values[:, 15:17]),
        ranges=np.ascontiguousarray(values[:, 9:13]),
    )


def readme_reconciliation(table: UwbTable) -> dict[str, object]:
    stated = README_COUNTS.get(table.scenario)
    return {
        "readme_stated": stated,
        "data_rows": table.n_rows,
        "readme_counts_header_line": stated == table.n_rows + 1,
        "matches_data_rows": stated == table.n_rows,
    }


def profile(table: UwbTable) -> PlanarProfile:
    sessions = []
    for sid in np.unique(table.session):
        idx = np.flatnonzero(table.session == sid)
        sessions.append(
            PlanarSession(
                scenario=table.scenario, session_id=int(sid), n_rows=len(idx), first_row=int(idx[0])
            )
        )
    return PlanarProfile(
        source_id=SOURCE_ID,
        member=table.member,
        member_sha256=table.member_sha256,
        adapter=f"{ADAPTER_ID}@{ADAPTER_VERSION}",
        units_note="no units declared by the depositor; values are consistent with metres in a "
        "~5 x 3 room but this is not stated and is not assumed",
        systems=[
            PlanarSystem(
                id="uwb",
                role="device_estimate",
                stage=COLUMN_STAGES["UWB"][0],
                x_column="UWB_x",
                y_column="UWB_y",
                meaning=COLUMN_STAGES["UWB"][1] + "; frame undeclared",
            ),
            PlanarSystem(
                id="optitrack",
                role="depositor_reference",
                stage=COLUMN_STAGES["OPT"][0],
                x_column="OPT_x",
                y_column="OPT_y",
                meaning=COLUMN_STAGES["OPT"][1] + "; frame and marker point undeclared; "
                "a separate measurement, not ground truth",
            ),
        ],
        sessions=sessions,
    )
