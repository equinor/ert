from __future__ import annotations

from importlib.resources import files
from signal import SIG_DFL, SIGINT, signal

from PyQt6.QtCore import QDir
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from everest.gui.main_window import EverestMainWindow


def run_gui(output_dir: str) -> None:
    # Replace Python's exception handler for SIGINT with the system default.
    #
    # Python's SIGINT handler is the one that raises KeyboardInterrupt. This is
    # okay normally (if a bit ugly), but when control is given to Qt this
    # exception handler will either get deadlocked because Python never gets
    # control back, or gets eaten by Qt because it ignores exceptions that
    # happen in Qt slots.
    signal(SIGINT, SIG_DFL)

    QDir.addSearchPath(
        "img", str(files("ert.gui").joinpath("../../ert/gui/resources/gui/img"))
    )

    app = QApplication(
        ["everest"]
    )  # Early so that QT is initialized before other imports
    app.setWindowIcon(QIcon("img:ert_icon.svg"))

    # Add arg parser if we are to pass more opts

    window = EverestMainWindow(output_dir)
    window.run()
    window.adjustSize()
    window.show()
    window.activateWindow()
    window.raise_()
    app.exec()
