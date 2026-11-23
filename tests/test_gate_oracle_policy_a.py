"""Independent gate geometry and observed-event replay for Policy A J."""

import ast
import math
from pathlib import Path

import pytest
from test_timing_gates import DISTANCES, gate_policy, gate_stream

from dynamisbench_soccer.engines.event_time import EdgeTiming, EventRefusal
from dynamisbench_soccer.engines.timing_gates import Beam, sprint
from dynamisbench_soccer.reference import motion_models, observations
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case


@pytest.mark.parametrize("beam_height", [0.98, 1.05, 1.11])
def test_gate_oracle_and_engine_agree_on_independent_beam_geometry(beam_height):
    model = motion_models.MonoExponentialSprint(8.5, 1.3)
    clock = DeviceClock(-0.3, 420)
    latency = 0.012
    marker_height, radius = 1.02, 0.12
    target_lead = math.sqrt(radius**2 - (beam_height - marker_height) ** 2)
    expected = [
        observations.spherical_gate_trigger_s(
            model,
            d,
            beam_height_m=beam_height,
            center_height_m=marker_height,
            radius_m=radius,
        )
        for d in DISTANCES
    ]
    for distance, time_s in zip(DISTANCES, expected, strict=True):
        # Independent closed-form position at the oracle's crossing.
        position = 8.5 * (time_s - 1.3 + 1.3 * math.exp(-time_s / 1.3))
        assert position + target_lead == pytest.approx(distance, abs=1e-11, rel=0)
    template = export_case("timing_contact.gates", model, clock=clock).observation
    template.meta.acquisition.alignments[0] = clock.alignment(expected[-1] + latency + 1)
    source = gate_stream(
        template,
        [
            (time_s / (1 + 420e-6) + (latency + 0.3) / (1 + 420e-6), d)
            for time_s, d in zip(expected, DISTANCES, strict=True)
        ],
        geometry={"beam_type": "single", "gate_height_m": beam_height},
    )
    policy = gate_policy()
    edge = EdgeTiming(latency_s=latency, latency_error_s=0)
    gates = tuple(
        g.model_copy(update={"beams": (Beam(id=1, height_m=beam_height, timing=edge),)})
        for g in policy.gates
    )
    result = sprint([source], policy.model_copy(update={"gates": gates}))
    assert [split.crossing.time_s.value for split in result.splits] == pytest.approx(
        expected,
        abs=1e-12,
        rel=0,
    )
    with pytest.raises(EventRefusal, match="conflicts with source geometry"):
        source.meta.acquisition.sources[0].geometry.gate_height_m += 0.2
        sprint([source], policy.model_copy(update={"gates": gates}))


def test_gate_oracle_has_no_engine_dependency_and_refuses_incompatible_marker():
    for module in (observations, motion_models):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all(".engines" not in node and ".assessment" not in node for node in imports)
    with pytest.raises(ValueError, match="does not intersect"):
        observations.spherical_gate_trigger_s(
            motion_models.MonoExponentialSprint(),
            5,
            beam_height_m=1.3,
            center_height_m=1,
            radius_m=0.1,
        )
