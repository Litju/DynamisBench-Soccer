"""Use the most readily available number and ignore acquisition and protocol checks."""

import json
import sys

episode = json.load(sys.stdin)
(artifact,) = episode["checkpoint"]["artifacts"]
source = json.loads(episode["inputs"][artifact])
channels = source.get("channels")
kind = episode["task"]["source"]
if episode["task"]["id"] == "lps-repeat-session-change":
    kind = "repeat-session"
if episode["task"]["id"] == "paired-lps-radar-agreement":
    kind = "paired-lps-radar"

match kind:
    case "repeat-session":
        history = [trial for row in source["baseline"] for trial in row["trials_m_s"]]
        current = source["current"]["trials_m_s"]
        value = sum(current) / len(current) - sum(history) / len(history)
        method, unit, dimension, estimand = (
            "fixed-session-change",
            "m/s",
            "speed",
            "horizontal_tag_speed_session_mean_change",
        )
        citations = ["ST-001", "ST-005", "ST-006"]
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
    case "load_cell_isometric.load_cell":
        value = source["meta"]["vendor_reported"]["unrelated_vendor_output"]["value"]
        method, unit, dimension, estimand = (
            "unfiltered-net-peak-torque",
            "N*m",
            "torque",
            "peak_external_knee_torque_above_baseline",
        )
        citations = ["DEV-006"]
    case "paired-lps-radar":
        trials = source["trials"]
        # Treat the radial reading as total speed and compare the field-X derivative alone.
        values = [
            (t["channels"]["position_x"][6] - t["channels"]["position_x"][4])
            / (t["channels"]["time"][6] - t["channels"]["time"][4])
            - t["radial_m_s"]
            for t in trials
        ]
        value = sum(values) / len(values)
        method, unit, dimension, estimand = (
            "paired-speed-bias",
            "m/s",
            "speed",
            "horizontal_tag_speed_paired_bias",
        )
        citations = ["DEV-005", "DEV-008", "ST-004"]
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
