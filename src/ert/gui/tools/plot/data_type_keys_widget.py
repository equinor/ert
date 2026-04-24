from PyQt6.QtCore import QSize
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import override

from ert.gui.ertwidgets import SearchBox
from ert.gui.icon_utils import load_icon
from ert.gui.utils import is_everest_application

from .data_type_keys_list_model import DataTypeKeysListModel, DataTypeSeparator
from .data_type_proxy_model import DataTypeProxyModel
from .plot_api import PlotApiKeyDefinition
from .widgets import FilterPopup

_EVEREST_GROUP_ORDER = [
    "everest_objectives",
    "everest_batch_objectives",
    "everest_constraints",
    "everest_parameters",
]

_EVEREST_GROUP_LABELS: dict[str, str] = {
    "everest_objectives": "Objectives",
    "everest_batch_objectives": "Aggregated objective values",
    "everest_constraints": "Constraints",
    "everest_parameters": "Controls",
}


def _group_everest_keys(
    key_defs: list[PlotApiKeyDefinition],
) -> list[PlotApiKeyDefinition | DataTypeSeparator]:
    groups: dict[str, list[PlotApiKeyDefinition]] = {}
    for key_def in key_defs:
        origin = key_def.metadata.get("data_origin", "")
        groups.setdefault(origin, []).append(key_def)

    ordered_origins = _EVEREST_GROUP_ORDER + [
        origin for origin in groups if origin not in _EVEREST_GROUP_ORDER
    ]

    result: list[PlotApiKeyDefinition | DataTypeSeparator] = []
    for origin in ordered_origins:
        if origin not in groups:
            continue
        label = _EVEREST_GROUP_LABELS.get(origin, origin)
        result.append(DataTypeSeparator(label=f"— {label} —"))
        result.extend(groups[origin])
    return result


class _LegendMarker(QWidget):
    """A widget that shows a colored box"""

    def __init__(self, color: QColor) -> None:
        QWidget.__init__(self)

        self.setMaximumSize(QSize(12, 12))
        self.setMinimumSize(QSize(12, 12))

        self.color = color

    @override
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

        is_everest = is_everest_application()
        model_items: list[PlotApiKeyDefinition | DataTypeSeparator] = (
            _group_everest_keys(key_defs) if is_everest else list(key_defs)
        )
        self.model = DataTypeKeysListModel(model_items)
        self.filter_model = DataTypeProxyModel(self, self.model)

        filter_layout = QHBoxLayout()

        self.search_box = SearchBox()
        self.search_box.filterChanged.connect(self.setSearchString)
        filter_layout.addWidget(self.search_box)

        filter_popup_button = QToolButton()
        filter_popup_button.setIcon(load_icon("filter_list.svg"))
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

        if not is_everest:
            layout.addWidget(
                _Legend(
                    "Observations available", DataTypeKeysListModel.HAS_OBSERVATIONS
                )
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
        for i in range(self.filter_model.rowCount()):
            # Need to skip the DataTypeSeparators
            index = self.filter_model.index(i, 0)
            source_index = self.filter_model.mapToSource(index)
            if self.model.itemAt(source_index) is not None:
                self.data_type_keys_widget.setCurrentIndex(index)
                return

    def setSearchString(self, filter_: str | None) -> None:
        self.filter_model.setFilterFixedString(filter_ or "")

    def showFilterPopup(self) -> None:
        self.__filter_popup.show()
