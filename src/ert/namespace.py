from __future__ import annotations

import argparse
from typing import Callable


class Namespace(argparse.Namespace):
    """
    ERT argument parser namespace
    """

    mode: str
    config: str
    database_url: str
    verbose: bool
    experimental_mode: bool
    logdir: str
    func: Callable[[Namespace], None]
