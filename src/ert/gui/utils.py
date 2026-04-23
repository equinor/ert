from PyQt6.QtWidgets import QApplication

LEGEND_THRESHOLD = 5


def is_everest_application() -> bool:
    return QApplication.applicationName().lower() == "everest"
