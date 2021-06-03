import os
import sys

import matplotlib

import ert_shared


def headless():
    return "DISPLAY" not in os.environ


if headless():
    matplotlib.use("Agg")
else:
    matplotlib.use("Qt5Agg")

__version__ = ert_shared.__version__
