from collections.abc import Callable
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
from .customdialog import CustomDialog
from .ensembleselector import EnsembleSelector
from .listeditbox import ListEditBox
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
from .searchbox import SearchBox
from .stringbox import StringBox
from .suggestor import Suggestor
from .textbox import TextBox


def showWaitCursorWhileWaiting(func: Callable[..., Any]) -> Callable[..., Any]:
    """A function decorator to show the wait cursor while the function is working."""

    def wrapper(*arg: Any) -> Any:
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            res = func(*arg)
            return res
        finally:
            QApplication.restoreOverrideCursor()

    return wrapper


__all__ = [
    "ActiveRealizationsModel",
    "AnalysisModuleEdit",
    "CheckList",
    "ClosableDialog",
    "CopyButton",
    "CopyableLabel",
    "CreateExperimentDialog",
    "CustomDialog",
    "EnsembleSelector",
    "ErtSummary",
    "ListEditBox",
    "PathChooser",
    "PathModel",
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
]
