from __future__ import annotations

import argparse
from typing import Callable, Optional

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
    experiment_name: Optional[str] = None
    func: Callable[[Namespace, Optional[ErtPluginManager]], None]
