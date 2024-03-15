from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErtConfig
from ert.enkf_main import sample_prior
from ert.gui.ertwidgets import showWaitCursorWhileWaiting
from ert.gui.ertwidgets.checklist import CheckList
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.ertwidgets.models.selectable_list_model import SelectableListModel
from ert.gui.ertwidgets.storage_info_widget import StorageInfoWidget
from ert.gui.ertwidgets.storage_widget import StorageWidget

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


class EnsembleInitializationConfigurationPanel(QTabWidget):
    @showWaitCursorWhileWaiting
    def __init__(self, config: ErtConfig, notifier, ensemble_size: int):
        QTabWidget.__init__(self)
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self.notifier = notifier
        self.setMinimumWidth(1200)
        self.addCreateNewEnsembleTab()
        self.addInitializeFromScratchTab()

    @property
    def storage(self):
        return self.notifier.storage

    def addCreateNewEnsembleTab(self):
        panel = QWidget()
        panel.setObjectName("create_new_ensemble_tab")

        layout = QGridLayout()
        storage_widget = StorageWidget(
            self.notifier, self.ert_config, self.ensemble_size
        )
        self._storage_info_widget = StorageInfoWidget()

        layout.addWidget(storage_widget, 0, 0)
        layout.addWidget(self._storage_info_widget, 0, 1)
        panel.setLayout(layout)

        storage_widget.onSelectExperiment.connect(
            self._storage_info_widget.setExperiment
        )
        storage_widget.onSelectEnsemble.connect(self._storage_info_widget.setEnsemble)

        self.addTab(panel, "Create new experiment")

    def addInitializeFromScratchTab(self):
        panel = QWidget()
        panel.setObjectName("initialize_from_scratch_panel")
        layout = QVBoxLayout()

        target_ensemble = EnsembleSelector(self.notifier, show_only_undefined=True)
        row = createRow(QLabel("Target ensemble:"), target_ensemble)
        layout.addLayout(row)

        check_list_layout, parameter_model, members_model = createCheckLists(
            self.ert_config.model_config.num_realizations,
            self.ert_config.ensemble_config.parameters,
        )
        layout.addLayout(check_list_layout)

        layout.addSpacing(10)

        initialize_button = QPushButton("Initialize")
        initialize_button.setObjectName("initialize_from_scratch_button")
        initialize_button.setMinimumWidth(75)
        initialize_button.setMaximumWidth(150)

        @showWaitCursorWhileWaiting
        def initializeFromScratch(_):
            parameters = parameter_model.getSelectedItems()
            sample_prior(
                ensemble=target_ensemble.currentData(),
                active_realizations=[int(i) for i in members_model.getSelectedItems()],
                parameters=parameters,
            )

        def update_button_state():
            initialize_button.setEnabled(target_ensemble.count() > 0)

        update_button_state()
        target_ensemble.ensemble_populated.connect(update_button_state)
        initialize_button.clicked.connect(initializeFromScratch)
        initialize_button.clicked.connect(
            lambda: self._storage_info_widget.setEnsemble(target_ensemble.currentData())
        )
        initialize_button.clicked.connect(target_ensemble.populate)

        layout.addWidget(initialize_button, 0, Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(10)

        panel.setLayout(layout)
        self.addTab(panel, "Initialize from scratch")
