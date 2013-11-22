from PyQt4.QtCore import QAbstractItemModel, QModelIndex, Qt, QVariant
from ert_gui.models.connectors.init import CaseList


class PlotCaseModel(QAbstractItemModel):

    def __init__(self):
        QAbstractItemModel.__init__(self)

    def index(self, row, column, parent=None, *args, **kwargs):
        return self.createIndex(row, column, parent)

    def parent(self, index=None):
        return QModelIndex()

    def rowCount(self, parent=None, *args, **kwargs):
        items = CaseList().getList()
        return len(items)

    def columnCount(self, QModelIndex_parent=None, *args, **kwargs):
        return 1



    def data(self, index, role=None):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            items = CaseList().getList()
            row = index.row()
            item = items[row]

            if role == Qt.DisplayRole:
                return item

        return QVariant()

    def itemAt(self, index):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            row = index.row()
            return CaseList().getList()[row]

        return None










