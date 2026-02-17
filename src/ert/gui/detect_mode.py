from typing import cast

from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QWidget


def is_high_contrast_mode() -> bool:
    app = cast(QWidget, QApplication.instance())
    return (
        app is not None
        and app.palette().color(QPalette.ColorRole.Window).lightness() > 245
    )


def is_dark_mode() -> bool:
    app = cast(QWidget, QApplication.instance())
    return app is not None and app.palette().base().color().value() < 70
