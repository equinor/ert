import os

import matplotlib as mpl

from .main import run_gui


def headless() -> bool:
    return "DISPLAY" not in os.environ


if headless():
    mpl.use("Agg")
else:
    mpl.use("QtAgg")


__all__ = ["run_gui"]
