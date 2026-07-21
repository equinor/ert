import logging

from PyQt6.QtCore import pyqtBoundSignal


def log_plot_option_usage_once(
    signal: pyqtBoundSignal, logger: logging.Logger, option_name: str
) -> None:
    def log_usage(*_args: object) -> None:
        logger.info("Plot sidebar option used: '%s'", option_name)
        signal.disconnect(log_usage)

    signal.connect(log_usage)
