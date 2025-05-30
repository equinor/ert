import warnings
from typing import Any

from PyQt6.QtCore import QObject


class PostSimulationWarning(Warning):
    """Warning category for GUI-specific messages"""


class QtWarningHandler(QObject):
    def __init__(
        self,
        accepted_categories: list[type[Warning]],
        post_simulation_warnings: list[str],
    ) -> None:
        super().__init__()
        self.post_simulation_warnings = post_simulation_warnings
        self.original_showwarning = warnings.showwarning
        self.accepted_categories = accepted_categories or []
        warnings.showwarning = self.custom_showwarning

    def custom_showwarning(
        self,
        message: Any,
        category: Any,
        filename: Any,
        lineno: Any,
        file: Any = None,
        line: Any = None,
    ) -> None:
        if category in self.accepted_categories:
            self.post_simulation_warnings.append(message)

    def restore(self) -> None:
        warnings.showwarning = self.original_showwarning
