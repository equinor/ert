import os

import matplotlib

import ert.shared


def headless() -> bool:
    return "DISPLAY" not in os.environ


if headless():
    matplotlib.use("Agg")
else:
    matplotlib.use("Qt5Agg")

__version__ = ert.shared.__version__
