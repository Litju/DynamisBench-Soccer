"""Shared clock-authority checks for native contact and gate engine entry points."""

import pytest
from test_timing_contact import policy as contact_policy
from test_timing_gates import gate_policy

from dynamisbench_soccer.engines.contact import contact
from dynamisbench_soccer.engines.event_time import EdgeTiming, EventRefusal
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
    with pytest.raises(EventRefusal, match="pulses contradict"):
        evaluate()
    alignment.offset_s -= 0.041
    source_clock = case.observation.meta.acquisition.clocks[1]
    source_clock.drift_ppm += 3
    with pytest.raises(EventRefusal, match="conflicts with clock declarations"):
        evaluate()
