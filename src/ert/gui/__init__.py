import os
from typing import cast

import matplotlib as mpl
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QWidget

import ert.shared


def headless() -> bool:
    return "DISPLAY" not in os.environ


if headless():
    mpl.use("Agg")
else:
    mpl.use("Qt5Agg")


def is_high_contrast_mode() -> bool:
    app = cast(QWidget, QApplication.instance())
    return app.palette().color(QPalette.ColorRole.Window).lightness() > 245


def is_dark_mode() -> bool:
    app = cast(QWidget, QApplication.instance())
    return app.palette().base().color().value() < 70


__version__ = ert.shared.__version__
