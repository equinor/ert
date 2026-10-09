from collections import Counter
from collections.abc import Iterable
from typing import override

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QHeaderView,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from ert.config.parameter_config import ParameterConfig
from ert.gui.ertwidgets.models.parameter_configuration_state_model import (
    ParameterConfigurationStateModel,
)

_COLUMN_HEADERS = ("strategy", "parameter type", "count")
_PARAMETER_TYPE_DISPLAY_NAMES = {
    "gen_kw": "GenKW",
    "field": "Field",
    "surface": "Surface",
}
_TOOLTIP = (
    "Each Field or Surface counts as one configuration, "
    "regardless of its number of grid cells."
)
_SummaryRow = tuple[str, str, str]


class UpdateStrategySummaryWidget(QTableWidget):
    def __init__(
        self,
        parameter_state: ParameterConfigurationStateModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._parameter_state = parameter_state
        self.setObjectName("update_strategy_summary_widget")
        self.setAccessibleName("Update strategy counts by parameter type")
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(_COLUMN_HEADERS)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlternatingRowColors(True)
        horizontal_header = self.horizontalHeader()
        assert horizontal_header is not None
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header_font = horizontal_header.font()
        header_font.setBold(True)
        horizontal_header.setFont(header_font)
        horizontal_header.setStyleSheet(
            "QHeaderView::section { border-bottom: 2px solid palette(mid); }"
        )
        vertical_header = self.verticalHeader()
        assert vertical_header is not None
        vertical_header.setVisible(False)
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.setToolTip(_TOOLTIP)
        parameter_state.changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        rows = _summarize_parameters(self._parameter_state.parameters)

        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                self.setItem(row_index, column_index, QTableWidgetItem(value))

        self.updateGeometry()

    @override
    def sizeHint(self) -> QSize:
        size = super().sizeHint()
        horizontal_header = self.horizontalHeader()
        vertical_header = self.verticalHeader()
        assert horizontal_header is not None
        assert vertical_header is not None
        return QSize(
            size.width(),
            horizontal_header.sizeHint().height()
            + vertical_header.length()
            + 2 * self.frameWidth(),
        )

    @override
    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


def _summarize_parameters(
    parameter_configs: Iterable[ParameterConfig],
) -> list[_SummaryRow]:
    counts = Counter((p.update_strategy, p.type) for p in parameter_configs)

    return sorted(
        (
            strategy.value.capitalize() if strategy is not None else "Non-updatable",
            _PARAMETER_TYPE_DISPLAY_NAMES.get(parameter_type, parameter_type),
            f"{count:,}",
        )
        for (strategy, parameter_type), count in counts.items()
    )
