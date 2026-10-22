from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication


@contextmanager
def wait_cursor() -> Generator[None, None, None]:
    """A context manager to show the wait cursor while the body is executing."""
    QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def showWaitCursorWhileWaiting(func: Callable[..., Any]) -> Callable[..., Any]:
    """A function decorator to show the wait cursor while the function is working."""

    def wrapper(*arg: Any) -> Any:
        with wait_cursor():
            return func(*arg)

    return wrapper
