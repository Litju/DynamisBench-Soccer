"""Verify emitted limitations on actual native reports and both CLI sensitivity paths."""

import json
from pathlib import Path

import pytest
from test_timing_gates import gate_stream
from test_timing_geometry_policy_a import _contact_fixture, _linear_gate_fixture

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.contact import contact
from dynamisbench_soccer.engines.event_time import EdgeTiming
from dynamisbench_soccer.engines.timing_gates import Beam, FixedOffset, TriggerStart, sprint
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.registry.models import load
from dynamisbench_soccer.signals import io

ROOT = Path(__file__).resolve().parents[1]
CONTACT_SOURCE = (
    "DEV-003: The tested mat's drop-jump contact times showed condition-dependent bias against "
    "force plate measurements. This comparison does not establish accuracy "
    "for other systems or metrics."
)
GATE_SOURCE = (
    "DEV-004: The studied single- and dual-beam methods give different timing observations. "
    "Beam height, trigger and voting settings have no universal correction."
)


def _published_sources():
    return {claim.id: claim.sources for claim in load(ROOT / "registry").claims}


@pytest.mark.parametrize(
    ("actuator", "definition"),
    [
        ("contact_mat", "Mat contact is defined by the configured switch response."),
        ("pressure_pad", "Pad contact is defined by the configured pressure switch response."),
        (
            "optical_contact_bar",
            "Optical contact is defined by interruption of the configured light path.",
        ),
    ],
)
def test_contact_emits_sourced_model_and_actuator_limits(actuator, definition):
    source, chosen = _contact_fixture(actuator, "active_low")
    result = contact(source, chosen)
    assert _published_sources()["DEV-003"] == ["tenelsen2019"]
    assert CONTACT_SOURCE in result.limitations
    assert definition in result.limitations
    text = " ".join(result.limitations)
    for concept in (
        "switching threshold",
        "mechanical takeoff",
        "nearest-tick",
        "ON/OFF delays",
        "missed transition",
        "equal centre-of-mass height",
        "empirical interchangeability",
    ):
        assert concept in text
    diagnostic = contact(
        source,
        chosen.model_copy(
            update={
                "ballistic_equal_com_height": False,
                "debounce_s": 0.005,
            }
        ),
    )
    assert diagnostic.flights[0].flight_time_height_m is None
    assert diagnostic.flights[0].flight_time_height_refusal
    assert "A positive refractory window" in " ".join(diagnostic.limitations)
    assert "Flight-time height and height-based RSI are withheld" in " ".join(
        diagnostic.limitations
    )


@pytest.mark.parametrize(
    ("start", "description"),
    [
        ("first_gate", "The first gate defines elapsed time zero; motion before it is unobserved."),
        ("fixed_offset", "A declared offset estimates the start-to-first-gate time."),
        (
            "pressure_pad",
            "Elapsed time starts from the measured and corrected pressure-pad release.",
        ),
        ("sound_cue", "Elapsed time starts from the measured and corrected sound cue."),
    ],
)
def test_gate_limits_follow_the_executed_start_definition(start, description):
    source, policy, crossing = _linear_gate_fixture(1.05, 8)
    if start == "fixed_offset":
        selection = FixedOffset(mode=start, offset_s=crossing[0], error_s=0)
        policy = policy.model_copy(update={"start": selection})
    elif start in {"pressure_pad", "sound_cue"}:
        rows = [(0.01, 0, 0, 0, 1 if start == "pressure_pad" else 2)]
        rows.extend(
            (t, d, i, 1, 0)
            for i, (t, d) in enumerate(zip(source["t_s"], source["distance_m"], strict=True), 1)
        )
        source = gate_stream(
            export_case("timing_contact.gates").observation, rows, geometry={"gate_height_m": 1.05}
        )
        policy = policy.model_copy(
            update={
                "start": TriggerStart(
                    mode=start,
                    position_m=0,
                    timing=EdgeTiming(latency_s=0.01, latency_error_s=0),
                )
            }
        )
    result = sprint([source], policy)
    assert _published_sources()["DEV-004"] == ["haugen2014"]
    assert GATE_SOURCE in result.limitations
    assert description in result.limitations
    text = " ".join(result.limitations)
    for concept in (
        "body part",
        "configured height",
        "nearest-tick",
        "link and trigger delays",
        "Instantaneous speed and radar",
        "Missing stations",
        "ambiguous beam groups",
        "empirical interchangeability",
    ):
        assert concept in text


def test_gate_limits_identify_voting_and_refractory_variant():
    source, policy, _ = _linear_gate_fixture(1.05, 8)
    rows = []
    for i, (time_s, position) in enumerate(
        zip(source["t_s"], source["distance_m"], strict=True), 1
    ):
        rows.extend([(time_s, position, i, 1), (time_s + 0.001, position, i, 2)])
    dual = gate_stream(
        export_case("timing_contact.gates").observation,
        rows,
        geometry={"gate_height_m": 1.05, "beam_type": "dual"},
    )
    gates = tuple(
        g.model_copy(
            update=dict(
                beam_type="dual",
                quorum=2,
                coincidence_s=0.005,
                beams=(*g.beams, Beam(id=2, height_m=1.1, timing=g.beams[0].timing)),
            )
        )
        for g in policy.gates
    )
    result = sprint([dual], policy.model_copy(update={"gates": gates, "debounce_s": 0.002}))
    assert "Multi-beam voting selects the quorum edge" in " ".join(result.limitations)
    assert "refractory window can suppress repeated interruptions" in " ".join(result.limitations)


@pytest.mark.parametrize("kind", ["contact", "gates"])
def test_cli_roundtrip_preserves_exact_limits_on_success_and_refusal(kind, tmp_path, capsys):
    if kind == "contact":
        source, chosen = _contact_fixture("pressure_pad", "active_low")
        valid = chosen.model_copy(update={"ballistic_equal_com_height": False})
        refused = chosen.model_copy(update={"initial_event": "airborne_snapshot"})

        def evaluate(method):
            return contact(source, method)

        base_text, variant_text = (
            CONTACT_SOURCE,
            "Flight-time height and height-based RSI are withheld",
        )
        refused_text = "initial contact state"
    else:
        source, chosen, crossing = _linear_gate_fixture(1.05, 8)
        valid = chosen.model_copy(
            update={
                "start": FixedOffset(
                    mode="fixed_offset",
                    offset_s=crossing[0],
                    error_s=0,
                )
            }
        )
        refused = chosen.model_copy(update={"max_segment_velocity_m_s": 2})

        def evaluate(method):
            return sprint([source], method)

        base_text, variant_text = GATE_SOURCE, "A declared offset estimates"
        refused_text = "implausible segment-average"
    bundle = tmp_path / "observation"
    io.write(source, bundle)
    files = []
    for name, method in (("primary", chosen), ("variant", valid), ("refused", refused)):
        path = tmp_path / f"{name}.json"
        path.write_text(method.model_dump_json())
        files.append(path)
    assert (
        main(
            [
                "timing",
                kind,
                str(bundle),
                "--policy",
                str(files[0]),
                "--sensitivity",
                str(files[1]),
                "--sensitivity",
                str(files[2]),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["sources"][0]["data_sha256"]
    assert report["sources"][0]["validated_metadata_sha256"]
    assert report["metrics"]["limitations"] == list(evaluate(chosen).limitations)
    variant, rejected = report["sensitivity"]
    assert variant["metrics"]["limitations"] == list(evaluate(valid).limitations)
    assert variant_text in " ".join(variant["metrics"]["limitations"])
    assert base_text in rejected["limitations"] and refused_text in rejected["refusal"]
    assert rejected["limitations"] == report["metrics"]["limitations"]
    assert "metrics" not in rejected
