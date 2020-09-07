import sys

if sys.version_info < (3,6):
    raise Exception("ERT GUI Python requires at least version 3.6 of Python")

import os
import matplotlib
import ert_shared

def headless():
    return "DISPLAY" not in os.environ

if headless():
    matplotlib.use("Agg")
else:
    matplotlib.use("Qt5Agg")

__version__ = ert_shared.__version__
