from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ert.config import ConfigValidationError
from ert.enkf_main import sample_prior, save_design_matrix_to_ensemble
from ert.gui.ertwidgets import (
    CheckList,
    EnsembleSelector,
    SelectableListModel,
    showWaitCursorWhileWaiting,
)

from .storage_info_widget import StorageInfoWidget
from .storage_widget import StorageWidget

if TYPE_CHECKING:
    from collections.abc import Collection

    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier


class ManageExperimentsPanel(QTabWidget):
    def __init__(
        self, config: ErtConfig, notifier: ErtNotifier, ensemble_size: int
    ) -> None:
        QTabWidget.__init__(self)
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self.notifier = notifier

        self._add_create_new_ensemble_tab()
        self._add_initialize_from_scratch_tab()

        self.installEventFilter(self)

        self.setWindowTitle("Manage experiments")
        self.setMinimumWidth(850)
        self.setMinimumHeight(250)

    def _add_create_new_ensemble_tab(self) -> None:
        panel = QWidget()
        panel.setObjectName("create_new_ensemble_tab")

        layout = QHBoxLayout()
        storage_widget = StorageWidget(
            self.notifier, self.ert_config, self.ensemble_size
        )
        self._storage_info_widget = StorageInfoWidget()

        layout.addWidget(storage_widget)
        layout.addWidget(self._storage_info_widget, stretch=1)
        panel.setLayout(layout)

        storage_widget.onSelectExperiment.connect(
            self._storage_info_widget.setExperiment
        )
        storage_widget.onSelectEnsemble.connect(self._storage_info_widget.setEnsemble)
        storage_widget.onSelectRealization.connect(
            self._storage_info_widget.setRealization
        )

        self.addTab(panel, "Create new experiment")

    def _add_initialize_from_scratch_tab(self) -> None:
        panel = QWidget()
        panel.setObjectName("initialize_from_scratch_panel")
        main_layout = QVBoxLayout()

        ensemble_layout = QHBoxLayout()
        ensemble_label = QLabel("Target ensemble:")
        ensemble_selector = EnsembleSelector(self.notifier, show_only_undefined=True)
        ensemble_selector.setMinimumWidth(300)
        ensemble_layout.addWidget(ensemble_label)
        ensemble_layout.addWidget(ensemble_selector)
        ensemble_layout.addStretch(1)

        main_layout.addLayout(ensemble_layout)

        center_layout = QHBoxLayout()
        design_matrix = self.ert_config.analysis_config.design_matrix
        parameters_config = self.ert_config.ensemble_config.parameter_configuration
        design_matrix_group = None
        realizations: Collection[int] = range(
            self.ert_config.runpath_config.num_realizations
        )
        if design_matrix is not None:
            try:
                parameters_config, design_matrix_group = (
                    design_matrix.merge_with_existing_parameters(parameters_config)
                )
                realizations = [
                    real
                    for real, active in enumerate(design_matrix.active_realizations)
                    if active
                ]
            except ConfigValidationError as exc:
                QMessageBox.warning(
                    self,
                    "Warning",
                    (
                        "The following issues were found when merging GenKW "
                        f'with design matrix parameters: "{exc}"'
                    ),
                )
                return

        parameter_model = SelectableListModel(
            [p.name for p in parameters_config] + [design_matrix_group.name]
            if design_matrix_group
            else self.ert_config.ensemble_config.parameters
        )
        parameter_check_list = CheckList(parameter_model, "Parameters")
        parameter_check_list.setMinimumWidth(500)
        center_layout.addWidget(parameter_check_list)

        members_model = SelectableListModel([str(member) for member in realizations])
        member_check_list = CheckList(members_model, "Members")
        center_layout.addWidget(member_check_list, stretch=1)

        main_layout.addLayout(center_layout)
        main_layout.addSpacing(10)

        initialize_button = QPushButton("Initialize")
        initialize_button.setObjectName("initialize_from_scratch_button")
        initialize_button.setMinimumWidth(75)
        initialize_button.setMaximumWidth(150)

        @showWaitCursorWhileWaiting
        def initialize_from_scratch(_: bool) -> None:
            parameters = parameter_model.getSelectedItems()
            active_realizations = [int(i) for i in members_model.getSelectedItems()]

            with self.notifier.write_storage() as storage:
                if (
                    design_matrix is not None
                    and design_matrix_group is not None
                    and design_matrix_group.name in parameters
                ):
                    parameters.remove(design_matrix_group.name)
                    save_design_matrix_to_ensemble(
                        design_matrix.design_matrix_df,
                        storage.get_ensemble(ensemble_selector.currentData()),
                        active_realizations,
                        design_group_name=design_matrix_group.name,
                    )
                sample_prior(
                    ensemble=storage.get_ensemble(ensemble_selector.currentData()),
                    active_realizations=active_realizations,
                    parameters=parameters,
                    random_seed=self.ert_config.random_seed,
                )

        @Slot()
        def update_button_state() -> None:
            if self.notifier.is_simulation_running:
                initialize_button.setEnabled(False)
            else:
                initialize_button.setEnabled(ensemble_selector.count() > 0)

        @Slot()
        def disableAdd() -> None:
            initialize_button.setEnabled(False)

        self.notifier.simulationStarted.connect(disableAdd)
        self.notifier.simulationEnded.connect(update_button_state)

        update_button_state()
        ensemble_selector.ensemble_populated.connect(update_button_state)
        initialize_button.clicked.connect(initialize_from_scratch)

        main_layout.addWidget(initialize_button, 1, Qt.AlignmentFlag.AlignRight)
        main_layout.addSpacing(10)
        panel.setLayout(main_layout)

        self.addTab(panel, "Initialize from scratch")

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is not None and a1.type() == QEvent.Type.Close:
            self.notifier.emitErtChange()
        return super().eventFilter(a0, a1)
