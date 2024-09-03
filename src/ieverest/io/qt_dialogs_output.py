from logging import CRITICAL, DEBUG, INFO, WARNING

from qtpy.QtWidgets import QMessageBox


class QtDialogsOut(object):
    """Support output requests through Qt Dialogs"""

    def __init__(self, qt_main_window, log_level=WARNING):
        super(QtDialogsOut, self).__init__()
        self._main_window = qt_main_window
        self.log_level = log_level

    def critical(self, message, force=False):
        if self.log_level <= CRITICAL or force:
            QMessageBox.critical(self._main_window, "Critical", message)

    def warning(self, message, force=False):
        if self.log_level <= WARNING or force:
            QMessageBox.warning(self._main_window, "Warning", message)

    def info(self, message, force=False):
        if self.log_level <= INFO or force:
            QMessageBox.information(self._main_window, "Information", message)

    def debug(self, message, force=False):
        if self.log_level <= DEBUG or force:
            QMessageBox.information(self._main_window, "Debug", message)
