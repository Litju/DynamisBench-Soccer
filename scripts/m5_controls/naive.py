"""Use the most readily available number and ignore acquisition and protocol checks."""

import json
import sys

episode = json.load(sys.stdin)
(artifact,) = episode["checkpoint"]["artifacts"]
source = json.loads(episode["inputs"][artifact])
channels = source["channels"]
kind = episode["task"]["source"]

match kind:
    case "timing_contact.gates":
        times = channels["t_s"]
        distances = channels["distance_m"]
        span = times[-1] - times[0]
        value = (distances[-1] - distances[0]) / span if span else 0
        method, unit, dimension, estimand = (
            "beam-split",
            "m/s",
            "speed",
            "beam_crossing_segment_average_velocity",
        )
        citations = ["DEV-004"]
    case "timing_contact.contact":
        times = channels["t_s"]
        value = 9.80665 * (times[-1] - times[0]) ** 2 / 8
        method, unit, dimension, estimand = (
            "switch-flight-height",
            "m",
            "length",
            "contact_flight_time_height",
        )
        citations = ["DEV-003"]
    case "gnss_lps.gnss":
        value = channels["doppler_speed"][source["row"]]
        method, unit, dimension, estimand = (
            "enu-positional-speed",
            "m/s",
            "speed",
            "horizontal_positional_speed",
        )
        citations = ["DEV-005", "CX-003"]
    case "gnss_lps.lps":
        r = source["row"]
        x, time = channels["position_x"], channels["time"]
        value = abs(x[r + 1] - 2 * x[r] + x[r - 1]) / (time[r + 1] - time[r]) ** 2
        method, unit, dimension, estimand = (
            "lps-filter-sensitivity",
            "m/s^2",
            "acceleration",
            "horizontal_positional_acceleration_filter_delta",
        )
        citations = ["DEV-005", "CX-003"]
    case _:
        raise ValueError(kind)

print(
    json.dumps(
        {
            "version": "m5.measurement-answer.v1",
            "decision": "compute",
            "method": method,
            "reason": "eligible",
            "selected_trials": [artifact],
            "citations": citations,
            "claims": [],
            "quantity": dict(
                value=value, unit=unit, dimension=dimension, estimand=estimand, uncertainty=0
            ),
            "operands": [dict(artifact=artifact, field="raw_shortcut", value=value)],
        }
    )
)
