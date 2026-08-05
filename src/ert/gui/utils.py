import logging

from PyQt6.QtCore import pyqtBoundSignal
from PyQt6.QtWidgets import QApplication

LEGEND_THRESHOLD = 5

# Number of significant digits to show in plots
SIGNIFICANT_DIGITS = 4

LONGEST_DEFAULT_EXPERIMENT_NAME = len("ensemble_information_filter")


def is_everest_application() -> bool:
    return QApplication.applicationName().lower() == "everest"


def truncate_string(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text

    truncation_indicator = "..."
    visible_length = max_length - len(truncation_indicator)
    front_len = visible_length // 2
    back_len = visible_length - front_len
    return f"{text[:front_len]}{truncation_indicator}{text[-back_len:]}"


def truncate_dropdown_item(text: str) -> str:
    return truncate_string(text, 100)


def truncate_experiment_name(name: str) -> str:
    return truncate_string(name, LONGEST_DEFAULT_EXPERIMENT_NAME)


def log_once(
    signal: pyqtBoundSignal,
    logger: logging.Logger,
    message: str,
    level: int = logging.INFO,
) -> None:
    """Log a message the first time a signal is emitted, then disconnect."""

    def log_and_disconnect(*_signal_args: object) -> None:
        logger.log(level, message)
        signal.disconnect(log_and_disconnect)

    signal.connect(log_and_disconnect)
