from typing import Dict, List

from qtpy.QtCore import QObject, Signal


class SelectableListModel(QObject):
    modelChanged = Signal()
    selectionChanged = Signal()

    def __init__(self, items: List[str]) -> None:
        QObject.__init__(self)
        self._selection: Dict[str, bool] = {}
        self._items = items

    def getList(self) -> List[str]:
        return self._items

    def isValueSelected(self, value: str) -> bool:
        return self._selection.get(value, True)

    def selectValue(self, value: str) -> None:
        self._setSelectState(value, True)
        self.selectionChanged.emit()

    def unselectValue(self, value: str) -> None:
        self._setSelectState(value, False)
        self.selectionChanged.emit()

    def unselectAll(self) -> None:
        for item in self.getList():
            self._setSelectState(item, False)

        self.selectionChanged.emit()

    def selectAll(self) -> None:
        for item in self.getList():
            self._setSelectState(item, True)

        self.selectionChanged.emit()

    def getSelectedItems(self) -> List[str]:
        return [item for item in self.getList() if self.isValueSelected(item)]

    def _setSelectState(self, key: str, state: bool) -> None:
        self._selection[key] = state
