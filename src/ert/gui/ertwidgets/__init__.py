# isort: skip_file
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication
from typing import Callable, Any


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


from .closabledialog import ClosableDialog
from .analysismoduleedit import AnalysisModuleEdit
from .searchbox import SearchBox
from .ensembleselector import EnsembleSelector
from .checklist import CheckList
from .stringbox import StringBox
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
from .message_box import ErtMessageBox
from .copy_button import CopyButton

__all__ = [
    "ActiveRealizationsModel",
    "AnalysisModuleEdit",
    "CheckList",
    "ClosableDialog",
    "CopyButton",
    "CopyableLabel",
    "CustomDialog",
    "EnsembleSelector",
    "ErtMessageBox",
    "ListEditBox",
    "PathChooser",
    "PathModel",
    "SearchBox",
    "SelectableListModel",
    "StringBox",
    "TargetEnsembleModel",
    "TextModel",
    "ValueModel",
    "showWaitCursorWhileWaiting",
]
