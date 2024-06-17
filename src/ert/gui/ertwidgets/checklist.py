from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets import SearchBox

if TYPE_CHECKING:
    from .models.selectable_list_model import SelectableListModel


class CheckList(QWidget):
    def __init__(
        self,
        model: SelectableListModel,
        label: str = "",
        custom_filter_button: Optional[QToolButton] = None,
    ):
        """
        :param custom_filter_button:  if needed, add a button that opens a
        custom filter menu. Useful when search alone isn't enough to filter the
        list.
        """
        QWidget.__init__(self)

        self._model = model

        layout = QVBoxLayout()

        self._createCheckButtons()

        self._list = QListWidget()
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self._search_box = SearchBox()

        check_button_layout = QHBoxLayout()

        check_button_layout.setContentsMargins(0, 0, 0, 0)
        check_button_layout.setSpacing(0)
        check_button_layout.addWidget(QLabel(label))
        check_button_layout.addStretch(1)
        check_button_layout.addWidget(self._checkAllButton)
        check_button_layout.addWidget(self._uncheckAllButton)

        layout.addLayout(check_button_layout)
        layout.addWidget(self._list)

        # Inserts the custom filter button, if provided. The caller is
        # responsible for all related actions.
        if custom_filter_button is not None:
            search_bar_layout = QHBoxLayout()
            search_bar_layout.addWidget(self._search_box)
            search_bar_layout.addWidget(custom_filter_button)
            layout.addLayout(search_bar_layout)
        else:
            layout.addWidget(self._search_box)

        self.setLayout(layout)

        self._checkAllButton.clicked.connect(self.checkAll)
        self._uncheckAllButton.clicked.connect(self.uncheckAll)
        self._list.itemChanged.connect(self.itemChanged)
        self._search_box.filterChanged.connect(self.filterList)
        self._list.customContextMenuRequested.connect(self.showContextMenu)

        self._model.selectionChanged.connect(self.modelChanged)
        self._model.modelChanged.connect(self.modelChanged)

        self.modelChanged()

    def _createCheckButtons(self) -> None:
        self._checkAllButton = QToolButton()
        self._checkAllButton.setIcon(QIcon("img:check.svg"))
        self._checkAllButton.setIconSize(QSize(16, 16))
        self._checkAllButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._checkAllButton.setAutoRaise(True)
        self._checkAllButton.setToolTip("Select all")
        self._uncheckAllButton = QToolButton()
        self._uncheckAllButton.setIcon(QIcon("img:checkbox_outline.svg"))
        self._uncheckAllButton.setIconSize(QSize(16, 16))
        self._uncheckAllButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._uncheckAllButton.setAutoRaise(True)
        self._uncheckAllButton.setToolTip("Unselect all")

    def itemChanged(self, item: QListWidgetItem) -> None:
        if item.checkState() == Qt.CheckState.Checked:
            self._model.selectValue(str(item.text()))
        elif item.checkState() == Qt.CheckState.Unchecked:
            self._model.unselectValue(str(item.text()))
        else:
            raise AssertionError("Unhandled checkstate!")

    def modelChanged(self) -> None:
        self._list.clear()

        items = self._model.getList()

        for item in items:
            list_item = QListWidgetItem(item)
            list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

            if self._model.isValueSelected(item):
                list_item.setCheckState(Qt.CheckState.Checked)
            else:
                list_item.setCheckState(Qt.CheckState.Unchecked)

            self._list.addItem(list_item)

        self.filterList(self._search_box.filter())

    def filterList(self, _filter: str) -> None:
        _filter = _filter.lower()

        for index in range(0, self._list.count()):
            item = self._list.item(index)
            assert item is not None
            text = item.text().lower()

            if not _filter or _filter in text:
                item.setHidden(False)
            else:
                item.setHidden(True)

    def checkAll(self) -> None:
        """
        Checks all visible items in the list.
        """
        for index in range(0, self._list.count()):
            item = self._list.item(index)
            assert item is not None
            if not item.isHidden():
                self._model.selectValue(str(item.text()))

    def uncheckAll(self) -> None:
        """
        Unchecks all items in the list, visible or not
        """
        self._model.unselectAll()

    def checkSelected(self) -> None:
        for item in self._list.selectedItems():
            self._model.selectValue(str(item.text()))

    def uncheckSelected(self) -> None:
        for item in self._list.selectedItems():
            self._model.unselectValue(str(item.text()))

    def showContextMenu(self, point: QPoint) -> None:
        p = self._list.mapToGlobal(point)
        menu = QMenu()
        check_selected = menu.addAction("Check selected")
        uncheck_selected = menu.addAction("Uncheck selected")
        menu.addSeparator()
        clear_selection = menu.addAction("Clear selection")

        selected_item = menu.exec_(p)

        if selected_item == check_selected:
            self.checkSelected()
        elif selected_item == uncheck_selected:
            self.uncheckSelected()
        elif selected_item == clear_selection:
            self._list.clearSelection()
