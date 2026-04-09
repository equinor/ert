from PyQt6.QtWidgets import QApplication


def is_everest_application() -> bool:
    return QApplication.applicationName().lower() == "everest"
