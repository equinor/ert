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