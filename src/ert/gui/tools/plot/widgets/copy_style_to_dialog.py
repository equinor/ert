from typing import Any, Dict, List, Optional

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QWidget,
)

from ert.gui.ertwidgets import CheckList
from ert.gui.tools.plot import FilterableKwListModel, FilterPopup
from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition


class CopyStyleToDialog(QDialog):
    def __init__(
        self,
        parent: Optional[QWidget],
        current_key: Any,
        key_defs: List[PlotApiKeyDefinition],
    ):
        QWidget.__init__(self, parent)
        self.setMinimumWidth(450)
        self.setMinimumHeight(200)
        self._dynamic = False
        self.setWindowTitle(f"Copy the style of {current_key} to other keys")
        self.activateWindow()

        layout = QFormLayout(self)

        self._filter_popup = FilterPopup(self, key_defs)
        self._filter_popup.filterSettingsChanged.connect(self.filterSettingsChanged)

        filter_popup_button = QToolButton()
        filter_popup_button.setIcon(QIcon("img:filter_list.svg"))
        filter_popup_button.clicked.connect(self._filter_popup.show)

        self._list_model = FilterableKwListModel(key_defs)
        self._list_model.unselectAll()

        self._cl = CheckList(self._list_model, custom_filter_button=filter_popup_button)

        layout.addWidget(self._cl)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.accept)
        apply_button.setDefault(True)

        close_button = QPushButton("Close")
        close_button.setToolTip("Hide this dialog")
        close_button.clicked.connect(self.reject)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(close_button)

        layout.addRow(button_layout)

    def getSelectedKeys(self) -> List[str]:
        return self._list_model.getSelectedItems()

    def filterSettingsChanged(self, item: Dict[str, bool]) -> None:
        for value, visible in item.items():
            self._list_model.setFilterOnMetadata("data_origin", value, visible)
        self._cl.modelChanged()
