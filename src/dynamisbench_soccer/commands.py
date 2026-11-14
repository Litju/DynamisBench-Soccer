"""Registry of CLI subcommand registrars.

Each registrar receives the ``argparse`` subparsers object and adds its commands.
Feature modules are imported here, in one place, to avoid import cycles with ``cli``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

Registrar = Callable[[argparse._SubParsersAction], None]  # type: ignore[type-arg]


def registrars() -> list[Registrar]:
    from dynamisbench_soccer.data import cli as data_cli
    from dynamisbench_soccer.registry import cli as registry_cli

    return [data_cli.register, registry_cli.register]
