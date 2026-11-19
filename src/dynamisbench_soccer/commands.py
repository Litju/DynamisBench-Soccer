"""Registry of CLI subcommand registrars.

Each registrar receives the ``argparse`` subparsers object and adds its commands.
Feature modules are imported here, in one place, to avoid import cycles with ``cli``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

Registrar = Callable[[argparse._SubParsersAction], None]  # type: ignore[type-arg]


def registrars() -> list[Registrar]:
    from dynamisbench_soccer.adapters import cli as adapters_cli
    from dynamisbench_soccer.assessment import cli as assess_cli
    from dynamisbench_soccer.data import cli as data_cli
    from dynamisbench_soccer.engines import cli as engines_cli
    from dynamisbench_soccer.engines import (
        gnss_lps_cli,
        imu_cli,
        load_cell_cli,
        lpt_cli,
        nordic_hip_cli,
        radar_laser_cli,
        timing_cli,
    )
    from dynamisbench_soccer.ontology import cli as ontology_cli
    from dynamisbench_soccer.qc import cli as qc_cli
    from dynamisbench_soccer.qualification import cli as qualification_cli
    from dynamisbench_soccer.registry import cli as registry_cli
    from dynamisbench_soccer.signals import cli as signals_cli

    return [
        data_cli.register,
        registry_cli.register,
        ontology_cli.register,
        signals_cli.register,
        adapters_cli.register,
        qc_cli.register,
        qualification_cli.register,
        gnss_lps_cli.register,
        radar_laser_cli.register,
        timing_cli.register,
        engines_cli.register,
        assess_cli.register,
        load_cell_cli.register,
        imu_cli.register,
        lpt_cli.register,
        nordic_hip_cli.register,
    ]
