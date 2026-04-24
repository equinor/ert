from dataclasses import dataclass
from typing import Any, overload

from PyQt6.QtCore import QAbstractItemModel, QModelIndex, QObject, Qt
from PyQt6.QtGui import QColor, QFont
from typing_extensions import override

from ert.gui.detect_mode import is_dark_mode
from ert.gui.icon_utils import load_icon

from .plot_api import PlotApiKeyDefinition


@dataclass(frozen=True)
class DataTypeSeparator:
    label: str


class DataTypeKeysListModel(QAbstractItemModel):
    DEFAULT_DATA_TYPE = QColor(255, 255, 255)
    HAS_OBSERVATIONS = QColor(237, 218, 116)
    GROUP_ITEM = QColor(64, 64, 64)

    def __init__(self, keys: list[PlotApiKeyDefinition | DataTypeSeparator]) -> None:
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
    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if index.isValid() and index.row() < len(self._keys):
            item = self._keys[index.row()]
            if isinstance(item, DataTypeSeparator):
                return Qt.ItemFlag.ItemIsEnabled
        else:
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    @override
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        assert isinstance(index, QModelIndex)

        if index.isValid():
            row = index.row()
            item = self._keys[row]

            if isinstance(item, DataTypeSeparator):
                if role == Qt.ItemDataRole.DisplayRole:
                    return item.label
                if role == Qt.ItemDataRole.FontRole:
                    font = QFont()
                    font.setBold(True)
                    return font
                if role == Qt.ItemDataRole.ForegroundRole:
                    return (
                        QColor(Qt.GlobalColor.white)
                        if is_dark_mode()
                        else QColor(Qt.GlobalColor.black)
                    )
                return None

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
            item = self._keys[row]
            if isinstance(item, DataTypeSeparator):
                return None
            return item

        return None
