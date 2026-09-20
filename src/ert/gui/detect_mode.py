from typing import cast

from PyQt6.QtWidgets import QApplication, QWidget


def is_dark_mode() -> bool:
    app = cast(QWidget, QApplication.instance())
    return app is not None and app.palette().base().color().value() < 70
