# isort: skip_file
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication
from typing import Any
from collections.abc import Callable


from .closabledialog import ClosableDialog
from .analysismoduleedit import AnalysisModuleEdit
from .searchbox import SearchBox
from .ensembleselector import EnsembleSelector
from .checklist import CheckList
from .stringbox import StringBox
from .textbox import TextBox
from .listeditbox import ListEditBox
from .customdialog import CustomDialog
from .pathchooser import PathChooser
from .models import (
    TextModel,
    ActiveRealizationsModel,
    TargetEnsembleModel,
    ValueModel,
    SelectableListModel,
    PathModel,
)
from .copyablelabel import CopyableLabel
from .copy_button import CopyButton


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
    "CustomDialog",
    "EnsembleSelector",
    "ListEditBox",
    "PathChooser",
    "PathModel",
    "SearchBox",
    "SelectableListModel",
    "StringBox",
    "TargetEnsembleModel",
    "TextBox",
    "TextModel",
    "ValueModel",
    "showWaitCursorWhileWaiting",
]
