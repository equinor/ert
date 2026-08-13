from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication


def showWaitCursorWhileWaiting(func: Callable[..., Any]) -> Callable[..., Any]:
    """A function decorator to show the wait cursor while the function is working."""

    def wrapper(*arg: Any) -> Any:
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            return func(*arg)
        finally:
            QApplication.restoreOverrideCursor()

    return wrapper
