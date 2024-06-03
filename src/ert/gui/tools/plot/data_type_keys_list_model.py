from typing import Any, List, Optional

from qtpy.QtCore import QAbstractItemModel, QModelIndex, Qt
from qtpy.QtGui import QColor, QIcon

from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition

default_index = QModelIndex()


class DataTypeKeysListModel(QAbstractItemModel):
    DEFAULT_DATA_TYPE = QColor(255, 255, 255)
    HAS_OBSERVATIONS = QColor(237, 218, 116)
    GROUP_ITEM = QColor(64, 64, 64)

    def __init__(self, keys: List[PlotApiKeyDefinition]):
        QAbstractItemModel.__init__(self)
        self._keys = keys
        self.__icon = QIcon("img:star_filled.svg")

    def index(
        self, row: int, column: int, parent: QModelIndex = default_index
    ) -> QModelIndex:
        return self.createIndex(row, column)

    def rowCount(self, parent: QModelIndex = default_index) -> int:
        return len(self._keys)

    def columnCount(self, parent: QModelIndex = default_index) -> int:
        return 1

    def data(self, index: QModelIndex, role: int = 0) -> Any:
        assert isinstance(index, QModelIndex)

        if index.isValid():
            items = self._keys
            row = index.row()
            item = items[row]

            if role == Qt.ItemDataRole.DisplayRole:
                return item.key
            elif role == Qt.ItemDataRole.BackgroundRole and item.observations:
                return self.HAS_OBSERVATIONS

        return None

    def itemAt(self, index: QModelIndex) -> Optional[PlotApiKeyDefinition]:
        assert isinstance(index, QModelIndex)

        if index.isValid():
            row = index.row()
            return self._keys[row]

        return None
