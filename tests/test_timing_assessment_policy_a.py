"""Native event eligibility is preserved across jump and gate-path assessment."""

import numpy as np
import pytest
from test_timing_contact import policy as contact_policy
from test_timing_gates import gate_policy

from dynamisbench_soccer.assessment.cod import CodSpec, assess_gates
from dynamisbench_soccer.assessment.jumps import contact_system_estimates
from dynamisbench_soccer.engines.contact import contact
from dynamisbench_soccer.engines.timing_gates import sprint
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import cmj, sj


@pytest.mark.parametrize("motion", [cmj(), sj()])
def test_native_contact_height_requires_engine_eligibility(motion):
    native = export_case("timing_contact.contact", motion).observation
    [height] = contact_system_estimates(contact(native, contact_policy()))
    takeoff, landing = motion.flight_intervals()[0]
    assert height.flight_time_s == pytest.approx(landing - takeoff, abs=1e-12, rel=0)
    expected = motion.state(takeoff).velocity_m_s ** 2 / (2 * motion.gravity_m_s2)
    assert height.height_flight_time_m == pytest.approx(expected, abs=1e-12, rel=0)
    gated = contact(native, contact_policy(ballistic_equal_com_height=False))
    assert gated.flights[0].flight_time_height_m is None
    with pytest.raises(ValueError, match="eligible ballistic flight"):
        contact_system_estimates(gated)


def test_native_gate_course_to_declared_path_assessment():
    case = export_case("timing_contact.gates")
    splits = sprint([case.observation], gate_policy()).splits
    spec = CodSpec(
        declared_path="synthetic declared path from the first beam",
        path_length_m=25,
        turn_distance_m=15,
        turn_angle_deg=90,
        source="radar",
    )
    start_position = splits[0].position_m
    path = [row.position_m - start_position for row in splits]
    cumulative = [row.cumulative_s.value for row in splits]
    result = assess_gates(path, cumulative, spec)
    independent = np.diff(case.truth["time"])
    assert result["total_time_s"] == pytest.approx(sum(independent), abs=1e-12, rel=0)
    assert [part["time_s"] for part in result["segments"]] == pytest.approx(
        independent, abs=1e-12, rel=0
    )
    assert result["entry_speed_m_s"] is None
    with pytest.raises(ValueError, match="full-course"):
        assess_gates(path[1:], cumulative[1:], spec)
    with pytest.raises(ValueError, match="last gate"):
        assess_gates(path[:-1], cumulative[:-1], spec)
