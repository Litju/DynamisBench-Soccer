"""WGS84 ellipsoidal/ECEF/ENU reference conversions, independent of device engines.

Constants: NGA WGS84. Conventions: PROJ cart and topocentric (EPSG 9602/9836).
Inverse latitude uses fixed-point iteration, limited to points near Earth's surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dynamisbench_soccer.signals.schema import FloatArray

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def geodetic_ecef(latitude_deg: float, longitude_deg: float, height_m: float) -> FloatArray:
    if (
        not all(math.isfinite(v) for v in (latitude_deg, longitude_deg, height_m))
        or not -90 <= latitude_deg <= 90
        or not -180 <= longitude_deg <= 180
    ):
        raise ValueError("finite WGS84 coordinates required")
    lat, lon = math.radians(latitude_deg), math.radians(longitude_deg)
    n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    return np.array(
        [
            (n + height_m) * math.cos(lat) * math.cos(lon),
            (n + height_m) * math.cos(lat) * math.sin(lon),
            (n * (1 - E2) + height_m) * math.sin(lat),
        ],
        dtype=np.float64,
    )


def ecef_geodetic(xyz: FloatArray) -> tuple[float, float, float]:
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)) or np.linalg.norm(xyz) < A / 2:
        raise ValueError("finite near-surface ECEF position required")
    x, y, z = map(float, xyz)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - E2))
    for _ in range(20):
        n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
        updated = math.atan2(z + E2 * n * math.sin(lat), p)
        if abs(updated - lat) < 1e-15:
            lat = updated
            break
        lat = updated
    else:
        raise ValueError("ECEF inverse failed to converge")
    height = p * math.cos(lat) + z * math.sin(lat) - A * math.sqrt(1 - E2 * math.sin(lat) ** 2)
    return math.degrees(lat), math.degrees(math.atan2(y, x)), height


@dataclass(frozen=True)
class LocalFrame:
    latitude_deg: float = 40
    longitude_deg: float = -3
    height_m: float = 0

    def __post_init__(self) -> None:
        geodetic_ecef(self.latitude_deg, self.longitude_deg, self.height_m)
        if abs(self.height_m) > 10000:
            raise ValueError("reference field origin must be within 10 km of ellipsoid")

    def enu_to_ecef_rotation(self) -> FloatArray:
        lat, lon = math.radians(self.latitude_deg), math.radians(self.longitude_deg)
        s, c = math.sin, math.cos
        return np.array(
            [
                [-s(lon), -s(lat) * c(lon), c(lat) * c(lon)],
                [c(lon), -s(lat) * s(lon), c(lat) * s(lon)],
                [0, c(lat), s(lat)],
            ],
            dtype=np.float64,
        )

    def to_geodetic(self, enu_m: FloatArray) -> tuple[float, float, float]:
        if enu_m.shape != (3,) or not np.all(np.isfinite(enu_m)) or np.linalg.norm(enu_m) > 10000:
            raise ValueError("finite ENU point within 10 km required")
        origin = geodetic_ecef(self.latitude_deg, self.longitude_deg, self.height_m)
        return ecef_geodetic(origin + self.enu_to_ecef_rotation() @ enu_m)

    def from_geodetic(
        self, latitude_deg: float, longitude_deg: float, height_m: float
    ) -> FloatArray:
        origin = geodetic_ecef(self.latitude_deg, self.longitude_deg, self.height_m)
        return self.enu_to_ecef_rotation().T @ (
            geodetic_ecef(latitude_deg, longitude_deg, height_m) - origin
        )
