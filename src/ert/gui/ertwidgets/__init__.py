# isort: skip_file
from pkg_resources import resource_filename
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor, QIcon, QMovie, QPixmap
from qtpy.QtWidgets import QApplication


def addHelpToWidget(widget, link):
    original_enter_event = widget.enterEvent

    def enterEvent(event):
        original_enter_event(event)

    widget.enterEvent = enterEvent


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


def resourceIcon(name):
    """Load an image as an icon"""
    return QIcon(resource_filename("ert.gui", "resources/gui/img/" + name))


def resourceStateIcon(on, off) -> QIcon:
    """Load two images as an icon with on and off states"""
    icon = QIcon()
    icon.addPixmap(resourceImage(on), state=QIcon.On)
    icon.addPixmap(resourceImage(off), state=QIcon.Off)
    return icon


def resourceImage(name) -> QPixmap:
    """Load an image as a Pixmap"""
    return QPixmap(resource_filename("ert.gui", "resources/gui/img/" + name))


def resourceMovie(name) -> QMovie:
    movie = QMovie(resource_filename("ert.gui", "resources/gui/img/" + name))
    movie.start()
    return movie


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
