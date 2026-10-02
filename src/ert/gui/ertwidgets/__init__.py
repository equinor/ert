from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication

from .analysismoduleedit import AnalysisModuleEdit
from .checklist import CheckList
from .closabledialog import ClosableDialog
from .copy_button import CopyButton
from .copyablelabel import CopyableLabel
from .create_experiment_dialog import CreateExperimentDialog
from .ensembleselector import EnsembleSelector
from .models import (
    ActiveRealizationsModel,
    ErtSummary,
    PathModel,
    SelectableListModel,
    TargetEnsembleModel,
    TextModel,
    ValueModel,
)
from .parameterviewer import get_parameters_button
from .pathchooser import PathChooser
from .search_bar import SearchBar
from .searchbox import SearchBox
from .stringbox import StringBox
from .suggestor import Suggestor
from .textbox import TextBox


@contextmanager
def wait_cursor() -> Iterator[None]:
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


__all__ = [
    "ActiveRealizationsModel",
    "AnalysisModuleEdit",
    "CheckList",
    "ClosableDialog",
    "CopyButton",
    "CopyableLabel",
    "CreateExperimentDialog",
    "EnsembleSelector",
    "ErtSummary",
    "PathChooser",
    "PathModel",
    "SearchBar",
    "SearchBox",
    "SelectableListModel",
    "StringBox",
    "Suggestor",
    "TargetEnsembleModel",
    "TextBox",
    "TextModel",
    "ValueModel",
    "get_parameters_button",
    "showWaitCursorWhileWaiting",
    "wait_cursor",
]
