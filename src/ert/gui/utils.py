from PyQt6.QtWidgets import QApplication

LEGEND_THRESHOLD = 5

# Number of significant digits to show in plots
SIGNIFICANT_DIGITS = 4


def is_everest_application() -> bool:
    return QApplication.applicationName().lower() == "everest"
