from typing import Any, overload

from PyQt6.QtCore import QAbstractItemModel, QModelIndex, QObject, Qt
from PyQt6.QtGui import QColor
from typing_extensions import override

from ert.gui.detect_mode import is_dark_mode
from ert.gui.icon_utils import load_icon

from .plot_api import PlotApiKeyDefinition


class DataTypeKeysListModel(QAbstractItemModel):
    DEFAULT_DATA_TYPE = QColor(255, 255, 255)
    HAS_OBSERVATIONS = QColor(237, 218, 116)
    GROUP_ITEM = QColor(64, 64, 64)

    def __init__(self, keys: list[PlotApiKeyDefinition]) -> None:
        QAbstractItemModel.__init__(self)
        self._keys = keys
        self.__icon = load_icon("star_filled.svg")

    @override
    def index(
        self, row: int, column: int, parent: QModelIndex | None = None
    ) -> QModelIndex:
        return self.createIndex(row, column)

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...
    @overload
    def parent(self) -> QObject: ...
    @override
    def parent(self, child: QModelIndex | None = None) -> QObject | QModelIndex:
        return QModelIndex()

    @override
    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return len(self._keys)

    @override
    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 1

    @override
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        assert isinstance(index, QModelIndex)

        if index.isValid():
            items = self._keys
            row = index.row()
            item = items[row]

            if role == Qt.ItemDataRole.DisplayRole:
                return item.key
            elif role == Qt.ItemDataRole.BackgroundRole and item.observations:
                return self.HAS_OBSERVATIONS
            elif (
                role == Qt.ItemDataRole.ForegroundRole
                and item.observations
                and is_dark_mode()
            ):
                return QColor(Qt.GlobalColor.black)

        return None

    def itemAt(self, index: QModelIndex) -> PlotApiKeyDefinition | None:
        assert isinstance(index, QModelIndex)

        if index.isValid():
            row = index.row()
            return self._keys[row]

        return None
