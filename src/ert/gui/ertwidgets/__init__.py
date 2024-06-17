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


# The following imports utilize the functions defined above:
from .legend import Legend  # noqa
from .validationsupport import ValidationSupport  # noqa
from .closabledialog import ClosableDialog  # noqa
from .analysismoduleedit import AnalysisModuleEdit  # noqa
from .activelabel import ActiveLabel  # noqa
from .searchbox import SearchBox  # noqa
from .ensembleselector import EnsembleSelector  # noqa
from .ensemblelist import EnsembleList  # noqa
from .checklist import CheckList  # noqa
from .stringbox import StringBox  # noqa
from .listeditbox import ListEditBox  # noqa
from .customdialog import CustomDialog  # noqa
from .summarypanel import SummaryPanel  # noqa
from .pathchooser import PathChooser  # noqa
from .models import TextModel  # noqa

__all__ = ["TextModel"]
