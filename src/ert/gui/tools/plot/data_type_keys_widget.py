from PyQt6.QtCore import QSize
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets import SearchBox

from .data_type_keys_list_model import DataTypeKeysListModel
from .data_type_proxy_model import DataTypeProxyModel
from .plot_api import PlotApiKeyDefinition
from .widgets import FilterPopup


class _LegendMarker(QWidget):
    """A widget that shows a colored box"""

    def __init__(self, color: QColor) -> None:
        QWidget.__init__(self)

        self.setMaximumSize(QSize(12, 12))
        self.setMinimumSize(QSize(12, 12))

        self.color = color

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)

        rect = self.contentsRect()
        rect.setWidth(rect.width() - 1)
        rect.setHeight(rect.height() - 1)
        painter.drawRect(rect)

        rect.setX(rect.x() + 1)
        rect.setY(rect.y() + 1)
        painter.fillRect(rect, self.color)


class _Legend(QWidget):
    """Combines a _LegendMarker with a label"""

    def __init__(self, legend: str | None, color: QColor) -> None:
        QWidget.__init__(self)

        self.setMinimumWidth(140)
        self.setMaximumHeight(25)

        self.legend = legend

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.legend_marker = _LegendMarker(color)
        self.legend_marker.setToolTip(legend or "")

        layout.addWidget(self.legend_marker)
        self.legend_label = QLabel(legend)
        layout.addWidget(self.legend_label)
        layout.addStretch()

        self.setLayout(layout)


class DataTypeKeysWidget(QWidget):
    dataTypeKeySelected = Signal()

    def __init__(self, key_defs: list[PlotApiKeyDefinition]) -> None:
        QWidget.__init__(self)

        self.__filter_popup = FilterPopup(self, key_defs)
        self.__filter_popup.filterSettingsChanged.connect(self.onItemChanged)

        layout = QVBoxLayout()

        self.model = DataTypeKeysListModel(key_defs)
        self.filter_model = DataTypeProxyModel(self, self.model)

        filter_layout = QHBoxLayout()

        self.search_box = SearchBox()
        self.search_box.filterChanged.connect(self.setSearchString)
        filter_layout.addWidget(self.search_box)

        filter_popup_button = QToolButton()
        filter_popup_button.setIcon(QIcon("img:filter_list.svg"))
        filter_popup_button.clicked.connect(self.showFilterPopup)
        filter_layout.addWidget(filter_popup_button)
        layout.addLayout(filter_layout)

        self.data_type_keys_widget = QListView()
        self.data_type_keys_widget.setModel(self.filter_model)
        self._sel_model = self.data_type_keys_widget.selectionModel()
        assert self._sel_model is not None
        self._sel_model.selectionChanged.connect(self.itemSelected)

        layout.addSpacing(15)
        layout.addWidget(self.data_type_keys_widget, 2)
        layout.addStretch()

        layout.addWidget(
            _Legend("Observations available", DataTypeKeysListModel.HAS_OBSERVATIONS)
        )

        self.setLayout(layout)

    def onItemChanged(self, item: dict[str, bool]) -> None:
        for value, visible in item.items():
            self.filter_model.setFilterOnMetadata("data_origin", value, visible)

    def itemSelected(self) -> None:
        selected_item = self.getSelectedItem()
        if selected_item is not None:
            self.dataTypeKeySelected.emit()

    def getSelectedItem(self) -> PlotApiKeyDefinition | None:
        index = self.data_type_keys_widget.currentIndex()
        source_index = self.filter_model.mapToSource(index)
        item = self.model.itemAt(source_index)
        return item

    def selectDefault(self) -> None:
        self.data_type_keys_widget.setCurrentIndex(self.filter_model.index(0, 0))

    def setSearchString(self, filter_: str | None) -> None:
        self.filter_model.setFilterFixedString(filter_ or "")

    def showFilterPopup(self) -> None:
        self.__filter_popup.show()
