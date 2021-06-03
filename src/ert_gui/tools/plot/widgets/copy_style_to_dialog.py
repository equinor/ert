from qtpy.QtWidgets import (
    QFormLayout,
    QWidget,
    QDialog,
    QPushButton,
    QHBoxLayout,
    QToolButton,
)

from ert_gui.ertwidgets import CheckList, resourceIcon
from ert_gui.tools.plot import FilterPopup, FilterableKwListModel
from ert_shared import ERT


class CopyStyleToDialog(QDialog):
    def __init__(self, parent, current_key, key_defs):
        QWidget.__init__(self, parent)
        self.setMinimumWidth(450)
        self.setMinimumHeight(200)
        self._dynamic = False
        self.setWindowTitle("Copy the style of {0} to other keys".format(current_key))
        self.activateWindow()

        layout = QFormLayout(self)

        self._filter_popup = FilterPopup(self, key_defs)
        self._filter_popup.filterSettingsChanged.connect(self.filterSettingsChanged)

        filter_popup_button = QToolButton()
        filter_popup_button.setIcon(resourceIcon("ide/cog_edit.png"))
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

    def getSelectedKeys(self):
        return self._list_model.getSelectedItems()

    def filterSettingsChanged(self, item):
        for value, visible in item.items():
            self._list_model.setFilterOnMetadata("data_origin", value, visible)
        self._cl.modelChanged()
