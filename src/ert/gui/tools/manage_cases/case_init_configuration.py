from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErtConfig
from ert.enkf_main import sample_prior
from ert.gui.ertwidgets import showWaitCursorWhileWaiting
from ert.gui.ertwidgets.caselist import CaseList
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.checklist import CheckList
from ert.gui.ertwidgets.models.selectable_list_model import SelectableListModel

if TYPE_CHECKING:
    from typing import List


def createCheckLists(ensemble_size: int, parameters: List[str]):
    parameter_model = SelectableListModel(parameters)

    parameter_check_list = CheckList(parameter_model, "Parameters")
    parameter_check_list.setMaximumWidth(300)

    members_model = SelectableListModel(
        [str(member) for member in range(ensemble_size)]
    )

    member_check_list = CheckList(members_model, "Members")
    member_check_list.setMaximumWidth(150)
    return (
        createRow(parameter_check_list, member_check_list),
        parameter_model,
        members_model,
    )


def createRow(*widgets):
    row = QHBoxLayout()

    for widget in widgets:
        row.addWidget(widget)

    row.addStretch()
    return row


class CaseInitializationConfigurationPanel(QTabWidget):
    @showWaitCursorWhileWaiting
    def __init__(self, config: ErtConfig, notifier, ensemble_size: int):
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self.notifier = notifier
        QTabWidget.__init__(self)
        self.setWindowTitle("Case management")
        self.setMinimumWidth(600)

        self.addCreateNewCaseTab()
        self.addInitializeFromScratchTab()
        self.addShowCaseInfo()
        self.currentChanged.connect(self.on_tab_changed)

    @property
    def storage(self):
        return self.notifier.storage

    def addCreateNewCaseTab(self):
        panel = QWidget()
        panel.setObjectName("create_new_case_tab")
        layout = QVBoxLayout()
        case_list = CaseList(self.ert_config, self.notifier, self.ensemble_size)

        layout.addWidget(case_list, stretch=1)

        panel.setLayout(layout)

        self.addTab(panel, "Create new case")

    def addInitializeFromScratchTab(self):
        panel = QWidget()
        panel.setObjectName("initialize_from_scratch_panel")
        layout = QVBoxLayout()

        target_case = CaseSelector(self.notifier)
        row = createRow(QLabel("Target case:"), target_case)
        layout.addLayout(row)

        check_list_layout, parameter_model, members_model = createCheckLists(
            self.ert_config.model_config.num_realizations,
            self.ert_config.ensemble_config.parameters,
        )
        layout.addLayout(check_list_layout)

        layout.addSpacing(10)

        initialize_button = QPushButton(
            "Initialize", objectName="initialize_from_scratch_button"
        )
        initialize_button.setMinimumWidth(75)
        initialize_button.setMaximumWidth(150)

        @showWaitCursorWhileWaiting
        def initializeFromScratch(_):
            parameters = parameter_model.getSelectedItems()
            target_ensemble = target_case.currentData()
            sample_prior(
                ensemble=target_ensemble,
                active_realizations=[int(i) for i in members_model.getSelectedItems()],
                parameters=parameters,
            )

        def update_button_state():
            initialize_button.setEnabled(target_case.count() > 0)

        update_button_state()
        target_case.case_populated.connect(update_button_state)
        initialize_button.clicked.connect(initializeFromScratch)
        layout.addWidget(initialize_button, 0, Qt.AlignCenter)

        layout.addSpacing(10)

        panel.setLayout(layout)
        self.addTab(panel, "Initialize from scratch")

    def addShowCaseInfo(self):
        case_widget = QWidget()
        layout = QVBoxLayout()

        case_selector = CaseSelector(
            self.notifier,
            update_ert=False,
        )
        row1 = createRow(QLabel("Select case:"), case_selector)

        layout.addLayout(row1)

        self._case_info_area = QTextEdit(objectName="html_text")
        self._case_info_area.setReadOnly(True)
        self._case_info_area.setMinimumHeight(300)

        row2 = createRow(QLabel("Case info:"), self._case_info_area)

        layout.addLayout(row2)

        case_widget.setLayout(layout)

        self.show_case_info_case_selector = case_selector
        case_selector.currentIndexChanged[int].connect(self._showInfoForCase)
        self.notifier.ertChanged.connect(self._showInfoForCase)

        self.addTab(case_widget, "Case info")

    def _showInfoForCase(self, index=None):
        if index is None:
            if self.notifier.current_case is not None:
                states = self.notifier.current_case.state_map
            else:
                states = []
        else:
            ensemble = self.show_case_info_case_selector.itemData(index)
            states = ensemble.state_map if ensemble is not None else []

        html = "<table>"
        for state_index, value in enumerate(states):
            html += f"<tr><td width=30>{state_index:d}.</td><td>{value.name}</td></tr>"

        html += "</table>"

        self._case_info_area.setHtml(html)

    @showWaitCursorWhileWaiting
    def on_tab_changed(self, p_int):
        if self.tabText(p_int) == "Case info":
            self._showInfoForCase()
