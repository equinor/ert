from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import override

from ert.gui.ertwidgets import (
    CheckList,
    EnsembleSelector,
    SelectableListModel,
    showWaitCursorWhileWaiting,
)
from ert.sample_prior import sample_prior
from ert.storage import Ensemble
from ert.storage.realization_storage_state import RealizationStorageState

from .storage_info_widget import StorageInfoWidget
from .storage_widget import StorageWidget

if TYPE_CHECKING:
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

        def show_only_undefined_filter(
            ensembles: Iterable[Ensemble],
        ) -> Iterable[Ensemble]:
            return (
                ensemble
                for ensemble in ensembles
                if all(
                    RealizationStorageState.UNDEFINED in e
                    for e in ensemble.get_ensemble_state()
                )
            )

        # only show initialized ensembles
        filters: list[Callable[[Iterable[Ensemble]], Iterable[Ensemble]]] = [
            show_only_undefined_filter
        ]

        ensemble_selector = EnsembleSelector(self.notifier, filters=filters)
        ensemble_selector.setMinimumWidth(300)
        ensemble_layout.addWidget(ensemble_label)
        ensemble_layout.addWidget(ensemble_selector)
        ensemble_layout.addStretch(1)

        main_layout.addLayout(ensemble_layout)

        center_layout = QHBoxLayout()
        parameters_config = self.ert_config.parameter_configurations_with_design_matrix
        realizations = [
            real
            for real, active in enumerate(self.ert_config.active_realizations)
            if active
        ]

        parameter_model = SelectableListModel([p.name for p in parameters_config])
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
                sample_prior(
                    ensemble=storage.get_ensemble(ensemble_selector.currentData()),
                    active_realizations=active_realizations,
                    parameters=parameters,
                    random_seed=self.ert_config.random_seed,
                    num_realizations=self.ert_config.runpath_config.num_realizations,
                    design_matrix_df=(
                        self.ert_config.analysis_config.design_matrix.design_matrix_df
                        if self.ert_config.analysis_config.design_matrix
                        else None
                    ),
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

    @override
    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is not None and a1.type() == QEvent.Type.Close:
            self.notifier.emitErtChange()
        return super().eventFilter(a0, a1)
