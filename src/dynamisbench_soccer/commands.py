"""Registry of optional CLI subcommand registrars.

Each registrar receives the ``argparse`` subparsers object and adds its commands.
Keeping this list in one place avoids import cycles between ``cli`` and features.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

Registrar = Callable[["argparse._SubParsersAction[argparse.ArgumentParser]"], None]

REGISTRARS: list[Registrar] = []
