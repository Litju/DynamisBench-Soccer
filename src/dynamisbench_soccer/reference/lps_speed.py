"""Independent analytical truth for the fixed-grid LPS speed slice."""

from math import hypot


def linear_tag_speed(east_m_s: float, north_m_s: float) -> float:
    return hypot(east_m_s, north_m_s)


def exact_grid_extreme_speed(
    east_m_s: float, north_m_s: float, sampling_hz: float, position_error_m: float
) -> float:
    """Opposite endpoint errors along the motion give the upper speed extreme."""
    return linear_tag_speed(east_m_s, north_m_s) + sampling_hz * position_error_m
