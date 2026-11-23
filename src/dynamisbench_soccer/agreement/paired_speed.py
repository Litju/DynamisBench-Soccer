"""Descriptive agreement for one ideal radar comparator and a native LPS row per trial."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from dynamisbench_soccer.agreement.stats import classify
from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import (
    AcquisitionDeclaration,
    AcquisitionRefusal,
)
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import Name, Nonnegative, Schema, Sha256
from dynamisbench_soccer.signals.schema import BundleMeta, DeviceClass, SignalBundle
from dynamisbench_soccer.statistics.reliability import Agreement, bland_altman

EAST_COSINE = 0.6
ROW = 5


class Trial(Schema):
    id: Name
    channels: dict[str, list[int | float]]
    channels_sha256: Sha256
    radial_m_s: Nonnegative


class PairedInput(Schema):
    meta: BundleMeta
    declaration: AcquisitionDeclaration
    trials: tuple[Trial, ...] = Field(min_length=3, max_length=8)
    projection: Literal["direction_corrected", "radial_only"]
    request: Literal["agreement", "reliability"] = "agreement"
    maximum_discrepancy_m_s: float = Field(gt=0)
    radial_standard_m_s: Nonnegative
    lps_standard_m_s: Nonnegative


@dataclass(frozen=True)
class PairedSpeed:
    classification: str
    pairs: tuple[tuple[str, float, float], ...] = ()
    agreement: Agreement | None = None
    expanded_uncertainty: float | None = None
    within_tolerance: bool | None = None


def compare(record: PairedInput) -> PairedSpeed:
    """Use one row per independent synthetic trial and the declared projection direction."""
    meta, declaration = record.meta, record.declaration
    if (
        meta.data_category is not DataCategory.SYNTHETIC
        or meta.device.device_class is not DeviceClass.LPS
        or meta.n_rows != 11
        or declaration.bundle_id != meta.bundle_id
        or declaration.native_hz != 25
        or declaration.timestamp_basis != "measurement"
        or declaration.latency_s
        or declaration.latency_uncertainty_s
    ):
        raise ValueError("paired reference requires native LPS and exact 25 Hz measurement time")
    lineage = digest(
        json.dumps(
            [trial.channels_sha256 for trial in record.trials], separators=(",", ":")
        ).encode()
    )
    if (
        meta.acquisition is None
        or len(meta.acquisition.sources) != 1
        or meta.acquisition.sources[0].clock != meta.acquisition.reference_clock
        or meta.provenance.source_sha256 != lineage
        or meta.acquisition.sources[0].source_sha256 != lineage
    ):
        raise ValueError("paired LPS requires a common clock and bound source lineage")
    if len({trial.id for trial in record.trials}) != len(record.trials) or len(
        {trial.channels_sha256 for trial in record.trials}
    ) != len(record.trials):
        return PairedSpeed("independent_pairs_required")

    pairs = []
    for trial in record.trials:
        raw = json.dumps(trial.channels, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if digest(raw.encode()) != trial.channels_sha256:
            raise ValueError("paired LPS channel digest differs from the source")
        channel_names = [channel.name for channel in meta.channels]
        if set(trial.channels) != set(channel_names):
            raise ValueError("paired LPS channels do not match the declaration")
        trial_meta = meta.model_copy(deep=True, update={"bundle_id": trial.id})
        if trial_meta.acquisition is None or len(trial_meta.acquisition.sources) != 1:
            raise ValueError("paired LPS requires one native source")
        trial_meta.acquisition.sources[0].source_sha256 = trial.channels_sha256
        trial_meta.provenance.source_sha256 = trial.channels_sha256
        bundle = SignalBundle(
            trial_meta,
            {name: np.asarray(trial.channels[name], dtype=np.float64) for name in channel_names},
        )
        try:
            lps = process(
                bundle,
                declaration.model_copy(update={"bundle_id": trial.id}),
                ProcessingOptions(thresholds=()),
            )
        except AcquisitionRefusal as exc:
            if str(exc).startswith(("signal integrity flag", "LPS anchor occlusion")):
                return PairedSpeed("failed_signal_qc")
            raise
        clock = lps.acquisition.clock
        if (
            not 0 < ROW < meta.n_rows - 1
            or clock.drift_ppm
            or clock.alignment_uncertainty_s
            or not np.allclose(np.diff(clock.elapsed_s), 0.04, rtol=0, atol=1e-12)
        ):
            raise ValueError("paired LPS requires an interior row on the exact uniform grid")
        steps = np.diff(lps.acquisition.position_m[:, :2], axis=0)
        steps /= np.diff(clock.elapsed_s)[:, None]
        vector = lps.kinematics.positional_velocity_m_s[ROW, :2]
        speed = float(lps.kinematics.horizontal_positional_speed_m_s[ROW])
        if speed <= 0 or not (
            np.allclose(vector / speed, (EAST_COSINE, 0.8), rtol=0, atol=1e-10)
            and np.allclose(steps, vector, rtol=0, atol=1e-10)
        ):
            raise ValueError("pair does not match the declared constant field direction")
        pairs.append((trial.id, trial.radial_m_s / EAST_COSINE, speed))

    verdict = classify(
        origin="analytical", definitions_match=record.projection == "direction_corrected"
    )
    if verdict.classification == "definition_mismatch":
        return PairedSpeed(verdict.classification)
    if record.request == "reliability":
        return PairedSpeed("repeat_sessions_required")
    agreement = bland_altman([(radar, lps) for _, radar, lps in pairs], missing="raise")
    # Each channel's calibration offset is shared across pairs, so it cannot average out.
    expanded = 2 * math.hypot(record.radial_standard_m_s / EAST_COSINE, record.lps_standard_m_s)
    if not math.isfinite(expanded):
        raise ValueError("invalid expanded uncertainty")
    within = max(map(abs, (*agreement.limits, agreement.bias))) <= record.maximum_discrepancy_m_s
    return PairedSpeed(verdict.classification, tuple(pairs), agreement, expanded, within)
