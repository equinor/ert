from qtpy.QtCore import Slot, Qt, QAbstractListModel, QModelIndex
from qtpy.QtGui import QClipboard
from qtpy.QtWidgets import QApplication


class FileModel(QAbstractListModel):
    def __init__(self, parent=None):
        super(FileModel, self).__init__(parent)
        self._rows = []
        self._new_line = True

    def rowCount(self, parent):
        """
        Overloaded Qt function. Return the number of rows in this model.
        :type parent: QModelIndex
        """
        return len(self._rows)

    def data(self, index, role=Qt.DisplayRole):
        """
        Overloaded Qt function. Return data for index.
        :type index: QModelIndex
        """
        if not index.isValid() or index.row() >= len(self._rows):
            return

        line = self._rows[index.row()]
        if role == Qt.DisplayRole:
            return line

    @Slot(str)
    def append_text(self, text):
        if not self._new_line:
            text = self._rows[-1] + text
        self._new_line = text[-2] == "\n"

        rows = text.splitlines(False)
        first = len(self._rows)
        last = first + len(rows)

        self.beginInsertRows(QModelIndex(), first, last)
        self._rows.extend(rows)
        self.endInsertRows()

    @Slot()
    def copy_all(self):
        """Copy the entire document into clipboard"""
        QApplication.clipboard().setText("\n".join(self._rows), QClipboard.Clipboard)
