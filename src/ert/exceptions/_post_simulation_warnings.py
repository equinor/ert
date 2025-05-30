import warnings

from PyQt6.QtCore import QObject


class PostSimulationWarning(Warning):
    """Warning category for GUI-specific messages"""


class QtWarningHandler(QObject):
    def __init__(self, accepted_categories=None, warning_funnel=None) -> None:
        super().__init__()
        self.warning_log = warning_funnel
        self.original_showwarning = warnings.showwarning
        self.accepted_categories = accepted_categories or []
        warnings.showwarning = self.custom_showwarning

    def custom_showwarning(
        self, message, category, filename, lineno, file=None, line=None
    ):
        if category in self.accepted_categories:
            self.warning_log.append(message)

    def restore(self):
        warnings.showwarning = self.original_showwarning
