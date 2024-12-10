from __future__ import annotations

import argparse
from collections.abc import Callable

from ert.plugins.plugin_manager import ErtPluginManager


class Namespace(argparse.Namespace):
    """
    ERT argument parser namespace
    """

    mode: str
    config: str
    verbose: bool
    experimental_mode: bool
    logdir: str
    experiment_name: str | None = None
    func: Callable[[Namespace, ErtPluginManager | None], None]
