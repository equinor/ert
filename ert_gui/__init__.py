import sys

REQUIRED_VERSION_HEX = 0x02070000


if sys.hexversion < REQUIRED_VERSION_HEX:
    raise Exception("ERT GUI Python requires at least version 2.7 of Python")

import os
import matplotlib

def headless():
    return "DISPLAY" not in os.environ

if headless():
    matplotlib.use("Agg")
else:
    matplotlib.use("Qt4Agg")

try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'
