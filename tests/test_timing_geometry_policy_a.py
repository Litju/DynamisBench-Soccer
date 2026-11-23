"""RES-698 native switching and course-geometry controls independent of timing engines."""

import math

import pytest
from test_timing_contact import CLOCK, policy, stream
from test_timing_gates import gate_policy, gate_stream

from dynamisbench_soccer.engines.contact import contact
from dynamisbench_soccer.engines.event_time import EdgeTiming, EventRefusal, Timing
from dynamisbench_soccer.engines.timing_gates import Beam, FixedOffset, sprint
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import dj
from dynamisbench_soccer.signals.schema import Calibration


def _prescribed_switch_pattern(polarity):
    on = int(polarity == "active_high")
    return [(0.109, on), (0.303, 1 - on), (0.729, on)]


def _contact_fixture(actuator, polarity):
    template = export_case("timing_contact.contact", dj()).observation
    source = stream(template, _prescribed_switch_pattern(polarity), shared=True)
    geometry = source.meta.acquisition.sources[0].geometry
    geometry.trigger_method = f"{actuator}:{polarity}"
    chosen = policy(
        system=actuator,
        initial_event="touchdown",
        timing=Timing(
            clock_model="shared",
            clocks={"reference": CLOCK},
            authority="prescribed independent switch and clock calibration",
        ),
        switch_on=EdgeTiming(latency_s=0.009, latency_error_s=0),
        switch_off=EdgeTiming(latency_s=0.003, latency_error_s=0),
    )
    return source, chosen


@pytest.mark.parametrize("actuator", ["contact_mat", "pressure_pad", "optical_contact_bar"])
@pytest.mark.parametrize("polarity", ["active_high", "active_low"])
def test_contact_actuator_polarity_and_edge_calibration(actuator, polarity):
    source, chosen = _contact_fixture(actuator, polarity)
    [flight] = contact(source, chosen).flights
    assert flight.preceding_touchdown.time_s.value == pytest.approx(0.1, abs=1e-12, rel=0)
    assert flight.takeoff.time_s.value == pytest.approx(0.3, abs=1e-12, rel=0)
    assert flight.touchdown.time_s.value == pytest.approx(0.72, abs=1e-12, rel=0)
    assert flight.ground_contact_s.value == pytest.approx(0.2, abs=1e-12, rel=0)
    assert flight.flight_s.value == pytest.approx(0.42, abs=1e-12, rel=0)
    assert flight.flight_s.low == pytest.approx(0.419, abs=1e-12, rel=0)
    assert flight.flight_s.high == pytest.approx(0.421, abs=1e-12, rel=0)


@pytest.mark.parametrize(
    ("declared", "reason"),
    [
        ("", "missing contact switching"),
        ("active_high", "actuator and switch polarity"),
        ("contact_mat", "actuator and switch polarity"),
        ("optical_contact_bar:active_high", "actuator and switch polarity"),
        ("contact_mat:active_high:active_low", "actuator and switch polarity"),
        ("contact_mat:active_low", "initial contact state"),
    ],
)
def test_contact_refuses_missing_ambiguous_or_contradictory_switches(declared, reason):
    source, chosen = _contact_fixture("contact_mat", "active_high")
    source.meta.acquisition.sources[0].geometry.trigger_method = declared
    with pytest.raises(EventRefusal, match=reason):
        contact(source, chosen)


def test_contact_refuses_bad_native_codes_timing_and_calibration():
    source, chosen = _contact_fixture("contact_mat", "active_low")
    source["contact"][1] = 0.5
    with pytest.raises(EventRefusal, match="binary"):
        contact(source, chosen)
    source["contact"][1] = 1
    source["t_s"][1] = 0.1
    with pytest.raises(EventRefusal, match="out-of-order"):
        contact(source, chosen)
    source["t_s"][1] = 0.303
    source.meta.acquisition.sources[0].calibration.append(
        Calibration(
            date="2000-01-01",
            method="unsupported numeric mapping",
            channel="contact",
            input_unit="1",
            output_unit="1",
            gain=-1,
            offset=1,
        )
    )
    with pytest.raises(EventRefusal, match="already calibrated"):
        contact(source, chosen)


def _sphere_crossings(positions, height, speed):
    lead = math.sqrt(0.12**2 - (height - 1.0) ** 2)
    return [(distance - lead) / speed for distance in positions]


def _linear_gate_fixture(height, speed):
    source = export_case("timing_contact.gates").observation
    positions = [4.0, 9.0, 16.0, 27.0]
    crossing = _sphere_crossings(positions, height, speed)
    times = [t + 0.017 for t in crossing]
    declared = gate_policy()
    gates = [
        gate.model_copy(
            update={
                "position_m": position,
                "beams": (
                    Beam(
                        id=1, height_m=height, timing=EdgeTiming(latency_s=0.017, latency_error_s=0)
                    ),
                ),
            }
        )
        for position, gate in zip(positions, declared.gates, strict=True)
    ]
    native = gate_stream(
        source,
        [(t, p, i, 1) for i, (t, p) in enumerate(zip(times, positions, strict=True), 1)],
        geometry={"gate_height_m": height},
    )
    return native, gate_policy(gates=gates), crossing


@pytest.mark.parametrize(("height", "speed"), [(0.93, 7.0), (1.05, 8.5)])
@pytest.mark.parametrize("fixed", [False, True])
def test_gate_applies_declared_route_and_analytical_beam_crossings(height, speed, fixed):
    source, chosen, crossing = _linear_gate_fixture(height, speed)
    if fixed:
        chosen = chosen.model_copy(
            update={
                "start": FixedOffset(
                    mode="fixed_offset", offset_s=crossing[0], error_s=0, start_position_m=0
                )
            }
        )
    result = sprint([source], chosen)
    assert result.estimand == "beam_crossing_segment_average_velocity"
    assert result.splits[0].segment_average_velocity_m_s is None
    for i, row in enumerate(result.splits):
        assert row.position_m == [4, 9, 16, 27][i]
        assert row.crossing.time_s.value == pytest.approx(crossing[i], rel=0, abs=1e-12)
        assert row.cumulative_s.value == pytest.approx(
            crossing[i] - (0 if fixed else crossing[0]),
            rel=0,
            abs=1e-12,
        )
        if i:
            assert row.split_s.value == pytest.approx(
                crossing[i] - crossing[i - 1], rel=0, abs=1e-12
            )
            assert row.segment_average_velocity_m_s.value == pytest.approx(speed, rel=0, abs=1e-11)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("height", "conflicts with source geometry"),
        ("unknown", "missing source beam geometry"),
        ("trigger", "missing source beam geometry"),
        ("distance", "unexpected gate ID or position"),
        ("fractional_id", "integer IDs"),
        ("repeated", "distinct IDs and increasing positions"),
        ("reversed", "distinct IDs and increasing positions"),
        ("infinite", "invalid gate course"),
    ],
)
def test_gate_refuses_unqualified_or_impossible_geometry(mutation, reason):
    source, chosen, _ = _linear_gate_fixture(1.05, 8)
    geometry = source.meta.acquisition.sources[0].geometry
    if mutation == "height":
        geometry.gate_height_m = 1.1
    elif mutation == "unknown":
        geometry.beam_type = "unknown"
    elif mutation == "trigger":
        geometry.trigger_method = None
        geometry.unknown.append("trigger_method")
    elif mutation == "distance":
        source["distance_m"][1] += 1
    elif mutation == "fractional_id":
        source["beam_id"][1] = 1.9
    else:
        position = {"repeated": 4, "reversed": 3, "infinite": float("inf")}[mutation]
        gates = [*chosen.gates]
        gates[1] = gates[1].model_copy(update={"position_m": position})
        chosen = chosen.model_copy(update={"gates": tuple(gates)})
    with pytest.raises(EventRefusal, match=reason):
        sprint([source], chosen)


def test_gate_refuses_a_first_gate_start_at_another_coordinate():
    source, chosen, _ = _linear_gate_fixture(1.05, 8)
    bad_start = chosen.start.model_copy(update={"start_position_m": 0})
    with pytest.raises(EventRefusal, match="must equal the first gate"):
        sprint([source], chosen.model_copy(update={"start": bad_start}))
