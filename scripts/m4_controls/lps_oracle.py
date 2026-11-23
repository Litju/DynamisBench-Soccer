"""Independent test solver for the fixed-grid LPS checkpoint contract."""

import json
import math
import sys


def answer(payload):
    (artifact,) = payload["checkpoint"]["artifacts"]
    document = json.loads(payload["inputs"][artifact])
    channels = document["channels"]
    row = document["row"]
    eligible = all(channels["signal_valid"])
    result = {
        "version": "m4.formal-answer.v1",
        "method": "three-point",
        "citations": ["DEV-005"],
        "claims": ["synthetic_tag_only"],
        "decision": "refuse",
        "reason": "not_eligible",
        "explanation": "A native position solution failed the validity check.",
        "selected_trials": [],
        "quantity": None,
        "operands": [],
    }
    if not eligible:
        return result
    before, after = row - 1, row + 1
    span = channels["time"][after] - channels["time"][before]
    x = channels["position_x"][after] - channels["position_x"][before]
    y = channels["position_y"][after] - channels["position_y"][before]
    result.update(
        decision="compute",
        reason="eligible",
        explanation="Norm of the centered horizontal derivative on the ideal grid.",
        selected_trials=[artifact],
        quantity={
            "value": math.hypot(x, y) / span,
            "unit": "m/s",
            "dimension": "speed",
            "estimand": "horizontal_positional_speed",
            "uncertainty": 2 * document["horizontal_error_m"] / span,
        },
        operands=[
            {"artifact": artifact, "field": f"{name}.{i}", "value": channels[name][i]}
            for name in ("time", "position_x", "position_y")
            for i in range(before, after + 1)
        ],
    )
    return result


if __name__ == "__main__":
    print(json.dumps(answer(json.load(sys.stdin))))
