"""Shared clock-authority checks for native contact and gate engine entry points."""

import pytest
from test_timing_contact import policy as contact_policy
from test_timing_gates import gate_policy

from dynamisbench_soccer.engines.contact import contact
from dynamisbench_soccer.engines.event_time import ClockTiming, EdgeTiming, EventRefusal, Timing
from dynamisbench_soccer.engines.timing_gates import sprint
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import cmj


@pytest.mark.parametrize("offset,drift", [(0.41, 751), (-0.23, -465)])
@pytest.mark.parametrize("kind", ["contact", "gates"])
def test_native_timing_entrypoints_enforce_clock_authority(kind, offset, drift):
    latency = 0.023
    edge = EdgeTiming(latency_s=latency, latency_error_s=0)
    motion = cmj() if kind == "contact" else None
    case = export_case(
        f"timing_contact.{kind}", motion, clock=DeviceClock(offset, drift), latency_s=latency
    )
    if kind == "contact":
        chosen = contact_policy(switch_on=edge, switch_off=edge)

        def evaluate():
            [flight] = contact(case.observation, chosen).flights
            return [flight.takeoff.time_s.value, flight.touchdown.time_s.value]

        expected = motion.flight_intervals()[0]
    else:
        chosen = gate_policy(edge)

        def evaluate():
            return [
                gate.crossing.time_s.value for gate in sprint([case.observation], chosen).splits
            ]

        expected = case.truth["time"]
    assert evaluate() == pytest.approx(expected, abs=1e-12, rel=0)
    alignment = case.observation.meta.acquisition.alignments[0]
    alignment.offset_s += 0.041
    with pytest.raises(EventRefusal, match="sync event disagrees"):
        evaluate()
    alignment.offset_s -= 0.041
    source_clock = case.observation.meta.acquisition.clocks[1]
    source_clock.drift_ppm += 3
    with pytest.raises(EventRefusal, match="conflicts with clock declarations"):
        evaluate()

    source_clock.drift_ppm -= 3
    acquisition = case.observation.meta.acquisition
    acquisition.reference_clock = alignment.clock
    alignment.reference_clock = alignment.clock
    with pytest.raises(EventRefusal, match="directly to reference_clock"):
        evaluate()


@pytest.mark.parametrize("kind", ["contact", "gates"])
def test_wireless_offset_resolution_and_edge_latency_have_distinct_units(kind):
    drift = 320
    edge_latency, link_latency = 0.014, 0.053
    tick, link_error, edge_error = 0.0002, 0.001, 0.002
    motion = cmj() if kind == "contact" else None
    case = export_case(
        f"timing_contact.{kind}",
        motion,
        clock=DeviceClock(0.48, drift),
        latency_s=edge_latency + link_latency,
    )
    timing = Timing(
        clock_model="wireless",
        authority="independent pulse and edge controls",
        clocks={
            "device": ClockTiming(
                resolution_s=tick, sync_latency_s=link_latency, sync_latency_error_s=link_error
            )
        },
    )
    edge = EdgeTiming(latency_s=edge_latency, latency_error_s=edge_error)
    if kind == "contact":
        [flight] = contact(
            case.observation,
            contact_policy(
                timing=timing,
                switch_on=edge,
                switch_off=edge,
            ),
        ).flights
        event = flight.takeoff
        expected = motion.flight_intervals()[0][0]
    else:
        event = sprint([case.observation], gate_policy(edge, timing=timing)).splits[0].crossing
        expected = float(case.truth["time"][0])
    width = (1 + drift * 1e-6) * tick / 2 + 1e-12 + link_error + edge_error
    assert event.time_s.value == pytest.approx(expected, rel=0, abs=1e-12)
    assert event.time_s.low == pytest.approx(expected - width, rel=0, abs=1e-12)
    assert event.time_s.high == pytest.approx(expected + width, rel=0, abs=1e-12)
