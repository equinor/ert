# isort: skip_file
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication


def showWaitCursorWhileWaiting(func):
    """A function decorator to show the wait cursor while the function is working."""

    def wrapper(*arg):
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
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
from .caseselector import CaseSelector  # noqa
from .caselist import CaseList  # noqa
from .checklist import CheckList  # noqa
from .stringbox import StringBox  # noqa
from .listeditbox import ListEditBox  # noqa
from .customdialog import CustomDialog  # noqa
from .summarypanel import SummaryPanel  # noqa
from .pathchooser import PathChooser  # noqa
from .suggestor_message import SuggestorMessage  # noqa
