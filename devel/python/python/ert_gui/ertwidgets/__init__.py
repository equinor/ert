from PyQt4 import QtGui, QtCore


def addHelpToWidget(widget, link):
    original_enter_event = widget.enterEvent

    def enterEvent(event):
        original_enter_event(event)
        try:
            from ert_gui.tools import HelpCenter
            HelpCenter.getHelpCenter("ERT").setHelpMessageLink(link)
        except AttributeError:
            pass

    widget.enterEvent = enterEvent


def showWaitCursorWhileWaiting(func):
    """A function decorator to show the wait cursor while the function is working."""
    def wrapper(*arg):
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        try:
            res = func(*arg)
            return res
        except:
            raise
        finally:
            QtGui.QApplication.restoreOverrideCursor()

    return wrapper

from .validationsupport import ValidationSupport
from .closabledialog import ClosableDialog
from .analysismoduleselector import AnalysisModuleSelector
from .activelabel import ActiveLabel
from .caseselector import CaseSelector
from .caselist import CaseList
from .checklist import CheckList
from .listeditbox import ListEditBox
from .customdialog import CustomDialog
from .summarypanel import SummaryPanel
