import os

import matplotlib as mpl

import ert.shared


def headless() -> bool:
    return "DISPLAY" not in os.environ


if headless():
    mpl.use("Agg")
else:
    mpl.use("QtAgg")

__version__ = ert.shared.__version__
