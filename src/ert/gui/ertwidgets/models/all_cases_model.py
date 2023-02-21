from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt

from ert.gui.ertnotifier import ErtNotifier


class AllCasesModel(QAbstractItemModel):
    def __init__(self, notifier: ErtNotifier):
        QAbstractItemModel.__init__(self)
        self._notifier = notifier

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
            case = items[row]

            if role == Qt.DisplayRole:
                return f"{case.name} - {case.started_at}"

        return None

    def itemAt(self, index):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            row = index.row()
            return self.getAllItems()[row]

        return None

    def getAllItems(self):
        all_case_list = list(self._notifier.storage.ensembles)
        return all_case_list

    def indexOf(self, item):
        items = self.getAllItems()

        if item in items:
            return items.index(item)

        return -1
