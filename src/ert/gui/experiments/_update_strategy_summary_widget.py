from collections import Counter
from collections.abc import Iterable
from typing import override

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QComboBox,
    QHeaderView,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from ert.config.parameter_config import (
    LocalizationType,
    ParameterConfig,
    ParameterType,
)
from ert.gui.ertwidgets.models.parameter_configuration_state_model import (
    ParameterConfigurationStateModel,
)

_ROW_HEIGHT = 32
_COLUMN_HEADERS = ("strategy", "parameter type", "count")
_PARAMETER_TYPE_DISPLAY_NAMES: dict[str, str] = {
    ParameterType.GEN_KW: "GenKW",
    ParameterType.FIELD: "Field",
    ParameterType.SURFACE: "Surface",
}
_TOOLTIP = (
    "Each Field or Surface counts as one configuration, "
    "regardless of its number of grid cells."
)
_PARAMETER_TYPES_BY_DISPLAY_NAME = {
    name: ParameterType(key) for key, name in _PARAMETER_TYPE_DISPLAY_NAMES.items()
}
_NOT_EDITABLE_TOOLTIP = "Update strategy is defined by the selected ensemble"
_SummaryRow = tuple[str, str, str]
_NON_UPDATABLE = "Non-updatable"


class UpdateStrategySummaryWidget(QTableWidget):
    def __init__(
        self,
        parameter_state: ParameterConfigurationStateModel,
        parent: QWidget | None = None,
        *,
        editable: bool = True,
    ) -> None:
        super().__init__(parent)
        self._parameter_state = parameter_state
        self._editable = editable
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
        self.setShowGrid(False)
        horizontal_header = self.horizontalHeader()
        assert horizontal_header is not None
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header_font = horizontal_header.font()
        header_font.setBold(True)
        horizontal_header.setFont(header_font)
        horizontal_header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        vertical_header = self.verticalHeader()
        assert vertical_header is not None
        vertical_header.setVisible(False)
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        vertical_header.setDefaultSectionSize(_ROW_HEIGHT)
        self.setToolTip(_TOOLTIP)
        parameter_state.changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        rows = _summarize_parameters(self._parameter_state.parameters)

        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                self.setItem(row_index, column_index, QTableWidgetItem(value))
            self._set_strategy_cell(row_index, row)

        self.updateGeometry()

    def _set_strategy_cell(self, row_index: int, row: _SummaryRow) -> None:
        strategy_name, type_name, _ = row
        parameter_type = _PARAMETER_TYPES_BY_DISPLAY_NAME.get(type_name)
        if parameter_type is None or strategy_name == _NON_UPDATABLE:
            self.removeCellWidget(row_index, 0)
            return
        if not self._editable or self._parameter_state.has_prior:
            self.removeCellWidget(row_index, 0)
            item = self.item(row_index, 0)
            assert item is not None
            item.setToolTip(_NOT_EDITABLE_TOOLTIP)
            return

        combobox = QComboBox()
        combobox.setObjectName(parameter_type)
        for localization_type in parameter_type.supported_localization_types:
            combobox.addItem(localization_type.value.capitalize(), localization_type)
        combobox.setCurrentIndex(combobox.findText(strategy_name))
        combobox.currentIndexChanged.connect(
            lambda index, box=combobox, ptype=parameter_type: self._on_strategy_chosen(
                ptype, box.itemData(index)
            )
        )
        self.setCellWidget(row_index, 0, combobox)

    def _on_strategy_chosen(
        self, parameter_type: ParameterType, strategy: LocalizationType
    ) -> None:
        self._parameter_state.apply_update_strategies(
            {**self._parameter_state.update_strategies, parameter_type: strategy}
        )

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
            strategy.value.capitalize() if strategy is not None else _NON_UPDATABLE,
            _PARAMETER_TYPE_DISPLAY_NAMES.get(parameter_type, parameter_type),
            f"{count:,}",
        )
        for (strategy, parameter_type), count in counts.items()
    )
