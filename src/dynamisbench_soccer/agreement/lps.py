"""LPS versus a closed-form radar projection of the same synthetic tag motion."""

from __future__ import annotations

import math

from dynamisbench_soccer.agreement.studies import Study, _native_evidence, _study
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import (
    AcquisitionDeclaration,
    AcquisitionRefusal,
)
from dynamisbench_soccer.signals.schema import DeviceClass, SignalBundle


def compare_lps_radar(
    observation: SignalBundle,
    declaration: AcquisitionDeclaration,
    velocities_m_s: tuple[float, float],
    beam_rad: float,
    *,
    correct_projection: bool,
    options: ProcessingOptions,
) -> Study:
    """Compare identical planar motions at interior samples of a constant-speed trial."""
    if observation.meta.device.device_class is not DeviceClass.LPS:
        raise AcquisitionRefusal("the positional input must be LPS")
    if not all(math.isfinite(x) for x in (*velocities_m_s, beam_rad)):
        raise ValueError("finite analytical velocity and beam direction required")
    if options.thresholds:
        raise ValueError("this study compares speed, without a threshold protocol")
    result = process(observation, declaration, options)
    vx, vy = velocities_m_s
    horizontal = math.hypot(vx, vy)
    radial = vx * math.cos(beam_rad) + vy * math.sin(beam_rad)
    cosine = radial / horizontal if horizontal else 0
    if correct_projection and abs(cosine) < 0.2:
        raise ValueError("beam does not resolve the direction of travel")
    radar = radial / cosine if correct_projection else radial
    # Ignore boundaries with incomplete filter or difference support.
    speed = result.kinematics.horizontal_positional_speed_m_s[4:-4]
    if len(speed) < 3:
        raise ValueError("at least three interior comparisons required")
    pairs = [(radar, float(value)) for value in speed]
    evidence = _native_evidence(
        "gnss_lps.lps",
        0,
        "admitted_by_engine_refusal_gate",
        "horizontal_positional_speed",
        "constant speed",
    )
    evidence["bundle_id"] = observation.meta.bundle_id
    evidence["source_sha256"] = result.acquisition.source_sha256
    return _study(
        "lps_vs_radar_constant_horizontal_speed",
        "radar_laser.radar",
        "gnss_lps.lps",
        "running_speed",
        execution_basis=("closed_form", "engine"),
        definitions={
            "radar": "beam projection corrected using declared direction of travel"
            if correct_projection
            else "signed radial velocity along the beam",
            "lps": "horizontal norm of position derivatives",
        },
        definitions_match=correct_projection,
        pairs=pairs,
        tolerance=1e-10,
        assumptions=[
            "one declared constant planar velocity; ideal synthetic positions and fixed radar beam",
            "no empirical ranging, survey, or alignment evidence",
        ],
        source_evidence=[evidence],
    )
