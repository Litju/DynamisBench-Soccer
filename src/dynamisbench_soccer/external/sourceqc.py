"""Source-level QC that runs before (and independent of) schema-v2 conversion.

When strict conversion is blocked (for ``shorts-laser``: no published session date)
the canonical bundle QC (:mod:`dynamisbench_soccer.qc`) cannot run, because it needs a
:class:`~dynamisbench_soccer.signals.schema.SignalBundle`. These checks apply the same
``sprint_fv`` rule semantics (QC-S03, QC-S04, QC-R01, QC-R02) to the parsed source
arrays, with parameters read from ``qc/rules.toml``, and add source-contract checks
(``SRC-*``). A test asserts parity with the bundle evaluators on a converted bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamisbench_soccer.external.shorts_laser import (
    PUBLISHED_RESAMPLED_HZ,
    SOURCE_COLUMNS,
    SourceTable,
)
from dynamisbench_soccer.qc.checks import runs
from dynamisbench_soccer.qc.spec import RuleSet, Severity, Status
from dynamisbench_soccer.signals.schema import FloatArray

MIRRORED_RULES = ("QC-S03", "QC-S04", "QC-R01", "QC-R02")
MAX_RATE_MISMATCH = 0.01


@dataclass(frozen=True)
class SourceCheck:
    rule_id: str
    name: str
    severity: Severity
    status: Status
    detail: str = ""
    values: dict[str, float] = field(default_factory=dict)
    basis: str = ""


def _channels(table: SourceTable) -> dict[str, FloatArray]:
    return {c.channel: table.columns[c.source_column] for c in SOURCE_COLUMNS}


def _fs(time: FloatArray) -> float | None:
    """Same rule as ``qc.checks._fs`` for a measured time column."""
    dt = np.diff(time)
    if len(dt) == 0 or np.any(dt <= 0) or not np.allclose(dt, np.median(dt), rtol=1e-6, atol=1e-12):
        return None
    return float(1 / np.median(dt))


def run(table: SourceTable, rules: RuleSet) -> list[SourceCheck]:
    ch = _channels(table)
    t = ch["time"]
    dt = np.diff(t)
    fs = _fs(t)
    out: list[SourceCheck] = []
    contract = "source contract (RES-720 adapter)"

    def add(
        rid: str,
        name: str,
        sev: Severity,
        ok: bool,
        detail: str = "",
        basis: str = contract,
        **values: float,
    ) -> None:
        out.append(
            SourceCheck(
                rid,
                name,
                sev,
                Status.PASS if ok else Status.FAIL,
                "" if ok else detail,
                dict(values),
                basis,
            )
        )

    n_bad = int(np.sum(dt <= 0))
    add(
        "SRC-01",
        "time_strictly_increasing",
        Severity.REJECT,
        n_bad == 0,
        f"{n_bad} non-increasing step(s)",
        n_nonincreasing=float(n_bad),
    )
    measured = float(1 / np.median(dt)) if len(dt) else 0.0
    mismatch = abs(measured - PUBLISHED_RESAMPLED_HZ) / PUBLISHED_RESAMPLED_HZ
    add(
        "SRC-02",
        "grid_matches_published_rate",
        Severity.REJECT,
        fs is not None and mismatch <= MAX_RATE_MISMATCH,
        f"grid {measured:.3f} Hz vs published {PUBLISHED_RESAMPLED_HZ:g} Hz or non-uniform",
        measured_hz=measured,
        mismatch=mismatch,
    )
    dd = np.diff(ch["distance"])
    n_dec = int(np.sum(dd < 0))
    add(
        "SRC-03",
        "distance_non_decreasing",
        Severity.FLAG,
        n_dec == 0,
        f"{n_dec} decreasing step(s), largest {float(-dd.min()):.4f} m",
        n_decreasing=float(n_dec),
    )
    for name in ("velocity", "velocity_vendor_unsmoothed"):
        vmin = float(ch[name].min())
        add(
            "SRC-04",
            f"{name}_non_negative",
            Severity.FLAG,
            vmin >= 0,
            f"{name} minimum {vmin:.4f} m/s",
            minimum=vmin,
        )

    rs = {r.id: r for r in rules.rule}
    basis = f"mirrors {{}} params from qc/rules.toml v{rules.version}"
    # QC-S03 clipping: identical to checks.clipping over non-time channels.
    p = rs["QC-S03"].params
    longest_run, worst = 0, ""
    for name, values in ch.items():
        if name == "time":
            continue
        top = values.max()
        if top <= 0:
            continue
        longest = max((e - s for s, e in runs(values == top)), default=0)
        if longest > longest_run:
            longest_run, worst = longest, name
    add(
        "QC-S03",
        "clipping",
        rs["QC-S03"].severity,
        longest_run < int(p["min_run"]),
        f"{worst}: {longest_run} consecutive samples at maximum",
        basis.format("QC-S03"),
        longest_run_at_max=float(longest_run),
    )
    # QC-S04 timestamp integrity.
    p = rs["QC-S04"].params
    gaps = int(np.sum(dt > p["max_gap_factor"] / PUBLISHED_RESAMPLED_HZ))
    add(
        "QC-S04",
        "timestamp_integrity",
        rs["QC-S04"].severity,
        n_bad == 0 and gaps == 0,
        f"{n_bad} non-increasing, {gaps} gap(s)",
        basis.format("QC-S04"),
        n_gaps=float(gaps),
        max_dt_ms=float(dt.max() * 1000),
    )
    # QC-R01 / QC-R02 need a uniform rate, as in qc.checks._fs.
    for rid in ("QC-R01", "QC-R02"):
        if fs is None:
            add(
                rid,
                rs[rid].name,
                rs[rid].severity,
                False,
                "timing-dependent QC requires uniform increasing timestamps",
                basis.format(rid),
            )
    if fs is not None:
        p = rs["QC-R01"].params
        n = max(1, round(p["start_window_s"] * fs))
        v0 = float(np.median(ch["velocity"][:n]))
        add(
            "QC-R01",
            "radar_start_captured",
            rs["QC-R01"].severity,
            v0 <= p["max_start_speed_m_s"],
            f"already moving at {v0:.2f} m/s at record start",
            basis.format("QC-R01"),
            start_speed=v0,
        )
        p = rs["QC-R02"].params
        for name in ("velocity", "velocity_vendor_unsmoothed"):
            accel = float(np.max(np.abs(np.diff(ch[name])))) * fs
            add(
                "QC-R02",
                f"radar_acceleration_spikes[{name}]",
                rs["QC-R02"].severity,
                accel <= p["max_abs_accel_m_s2"],
                f"sample-to-sample {accel:.1f} m/s^2",
                basis.format("QC-R02"),
                max_accel=accel,
            )
    return out


def verdict(checks: list[SourceCheck]) -> str:
    failed = [c for c in checks if c.status is Status.FAIL]
    if any(c.severity is Severity.REJECT for c in failed):
        return "reject"
    return "accept_with_flags" if failed else "accept"
