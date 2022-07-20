from qtpy.QtCore import Qt, QAbstractItemModel, QModelIndex

from ert.gui.ertwidgets.models.ertmodel import getAllCases
from ert_shared.libres_facade import LibresFacade


class AllCasesModel(QAbstractItemModel):
    def __init__(self, facade: LibresFacade):
        self.facade = facade
        QAbstractItemModel.__init__(self)
        self.__data = []

    def index(self, row, column, parent=None, *args, **kwargs):
        return self.createIndex(row, column)

    def parent(self, index=None):
        return QModelIndex()

    def rowCount(self, parent=None, *args, **kwargs):
        items = self.getAllItems()
        return len(items)

    def columnCount(self, QModelIndex_parent=None, *args, **kwargs):
        return 1

    def data(self, index, role=None):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            items = self.getAllItems()
            row = index.row()
            item = items[row]

            if role == Qt.DisplayRole:
                return item

    def itemAt(self, index):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            row = index.row()
            return self.getAllItems()[row]

        return None

    def getAllItems(self):
        return getAllCases(self.facade)

    def indexOf(self, item):
        items = self.getAllItems()

        if item in items:

            return items.index(item)

        return -1
