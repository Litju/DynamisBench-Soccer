"""Typed, source-bound handoff from engine reports to assessment (RES-673).

An assessment never trusts a caller-supplied source label. The instrument is identified from
the engine report itself; a declared source that disagrees is refused. Radar, laser, GNSS
Doppler, GNSS position-derived and LPS position-derived speed are different estimands and
keep their identity, native/vendor stage, clock origin and provenance. Speed is never made
nonnegative by taking an absolute value: a trace with reversed sign is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dynamisbench_soccer.signals.schema import FloatArray

NEGATIVE_SPEED_TOLERANCE_M_S = 0.05  # same bound the radar/laser engine uses for sign integrity
NATIVE_STAGES = frozenset({"native"})
_REJECT_WORDS = frozenset({"reject", "rejected", "fail", "failed"})


class HandoffRefusal(ValueError):
    """The engine report cannot be used for the requested assessment."""


@dataclass(frozen=True)
class SpeedHandoff:
    source: str
    device: str
    time_s: FloatArray
    speed_m_s: FloatArray
    stage: str
    estimand: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def input_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "device": self.device,
            "stage": self.stage,
            "estimand": self.estimand,
            "n_samples": len(self.time_s),
            "provenance": self.provenance,
        }


def _arrays(time: object, speed: object) -> tuple[FloatArray, FloatArray]:
    try:
        t = np.asarray(time, dtype=np.float64)
        v = np.asarray(speed, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HandoffRefusal("time and speed must be numeric arrays") from exc
    if t.ndim != 1 or t.shape != v.shape or len(t) < 3:
        raise HandoffRefusal("time and speed must be equal-length 1-D arrays of >= 3 samples")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
        raise HandoffRefusal("time and speed must be finite")
    if not np.all(np.diff(t) > 0):
        raise HandoffRefusal("time must be strictly increasing")
    if float(v.min()) < -NEGATIVE_SPEED_TOLERANCE_M_S:
        raise HandoffRefusal(
            "negative speed: sign or beam direction unresolved; absolute value is never applied"
        )
    return t, v


def _check_verdict(report: dict[str, Any]) -> None:
    for key in ("verdict", "qc_verdict"):
        if str(report.get(key, "")).lower() in _REJECT_WORDS:
            raise HandoffRefusal(f"engine report carries a {report[key]!r} QC verdict")
    if report.get("diagnostic_only") is True:
        raise HandoffRefusal("engine report is diagnostic-only and cannot feed accepted metrics")


def _stage_ok(stage: str, allow_vendor_processed: bool) -> None:
    if stage not in NATIVE_STAGES and not allow_vendor_processed:
        raise HandoffRefusal(
            f"input stage {stage!r} is not native; vendor-processed data needs explicit opt-in"
        )


def _radar_laser(report: dict[str, Any], allow_vendor: bool) -> SpeedHandoff:
    quantity = report.get("native_quantity")
    device = {"velocity": "radar", "distance": "laser"}.get(str(quantity))
    if device is None:
        raise HandoffRefusal(f"unrecognised radar/laser native quantity {quantity!r}")
    stage = str(report.get("source_stage", ""))
    if not stage:
        raise HandoffRefusal("radar/laser report lacks source_stage")
    _stage_ok(stage, allow_vendor)
    t, v = _arrays(report.get("time_from_start_s"), report.get("velocity_m_s"))
    provenance = {
        k: report.get(k)
        for k in (
            "source_id",
            "native_channel",
            "filter_method",
            "derivative_method",
            "correction_method",
            "start_method",
            "start_reference_s",
            "nominal_hz",
            "observed_hz",
            "aiming_point",
            "mounting_height_m",
            "mounting_distance_behind_start_m",
        )
    }
    return SpeedHandoff(
        source=device,
        device=device,
        time_s=t,
        speed_m_s=v,
        stage=stage,
        estimand=str(report.get("estimand", "")),
        provenance={**provenance, "time_origin": "declared start of the radar/laser engine"},
    )


def _track(report: dict[str, Any], declared: str, allow_vendor: bool) -> SpeedHandoff:
    try:
        acq = report["acquisition"]
        kin = report["kinematics"]
        device = str(acq["device"])
        declaration = acq["declaration"]
        clock = acq["clock"]
        elapsed = clock["elapsed_s"]
    except (KeyError, TypeError) as exc:
        raise HandoffRefusal(f"malformed GNSS/LPS report: missing {exc}") from exc
    if device not in {"gnss", "lps"}:
        raise HandoffRefusal(f"unrecognised track device {device!r}")
    stage = str(declaration.get("stage", ""))
    _stage_ok(stage, allow_vendor)
    if declared == "gnss_doppler":
        if device != "gnss":
            raise HandoffRefusal(f"declared gnss_doppler but report is from {device}")
        speed = kin.get("doppler_speed_m_s")
        if speed is None:
            raise HandoffRefusal("report carries no Doppler speed")
        source, estimand = "gnss_doppler", "GNSS Doppler speed over ground"
    elif declared in {"gnss_position", "lps_position"}:
        if declared != f"{device}_position":
            raise HandoffRefusal(f"declared {declared} but report is from {device}")
        speed = kin.get("horizontal_positional_speed_m_s")
        source = declared
        estimand = f"horizontal speed differentiated from {device.upper()} positions"
    else:
        raise HandoffRefusal(f"declared source {declared!r} is not a GNSS/LPS speed")
    t, v = _arrays(elapsed, speed)
    return SpeedHandoff(
        source=source,
        device=device,
        time_s=t,
        speed_m_s=v,
        stage=stage,
        estimand=estimand,
        provenance={
            "source_id": acq.get("source_id"),
            "source_format": acq.get("source_format"),
            "data_sha256": acq.get("data_sha256"),
            "bundle_id": declaration.get("bundle_id"),
            "frame_id": acq.get("frame_id"),
            "sensor_placement": acq.get("sensor_placement"),
            "source_clock": clock.get("source_clock"),
            "reference_clock": clock.get("reference_clock"),
            "alignment_uncertainty_s": clock.get("alignment_uncertainty_s"),
            "time_origin": "elapsed reference time from the first admitted sample",
            "position_filter": report.get("options", {}).get("position_filter"),
            "doppler_filter": report.get("options", {}).get("doppler_filter"),
            "derivative": report.get("options", {}).get("derivative"),
        },
    )


def load_speed_report(
    path: Path, *, declared_source: str, allow_vendor_processed: bool = False
) -> SpeedHandoff:
    """Parse a radar/laser or GNSS/LPS engine JSON report into a source-bound speed trace."""
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HandoffRefusal(f"report is not valid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise HandoffRefusal("report must be a JSON object")
    _check_verdict(report)
    if "acquisition" in report and "kinematics" in report:
        handoff = _track(report, declared_source, allow_vendor_processed)
    elif "velocity_m_s" in report and "native_quantity" in report:
        handoff = _radar_laser(report, allow_vendor_processed)
    else:
        raise HandoffRefusal("not a recognised radar/laser or GNSS/LPS engine report")
    if handoff.source != declared_source:
        raise HandoffRefusal(
            f"declared source {declared_source!r} but the report identifies {handoff.source!r}"
        )
    return handoff
