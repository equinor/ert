import os
import sys

import matplotlib

import ert.shared


def headless():
    return "DISPLAY" not in os.environ


if headless():
    matplotlib.use("Agg")
else:
    matplotlib.use("Qt5Agg")


# Ensure that qtpy uses the correct Qt bindings
if "QT_API" not in os.environ:
    if sys.version_info > (3, 8):
        os.environ["QT_API"] = "pyqt6"

    import qtpy

    if sys.version_info > (3, 8):
        assert qtpy.PYQT6
    else:
        assert qtpy.PYQT5


__version__ = ert.shared.__version__
