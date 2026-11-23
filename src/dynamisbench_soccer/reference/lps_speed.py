"""Independent analytical truth for the fixed-grid LPS speed slice."""

from math import hypot


def linear_tag_speed(east_m_s: float, north_m_s: float) -> float:
    return hypot(east_m_s, north_m_s)
