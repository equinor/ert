from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ert.enkf_main import sample_prior
from ert.gui.ertwidgets import showWaitCursorWhileWaiting
from ert.gui.ertwidgets.checklist import CheckList
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.ertwidgets.models.selectable_list_model import SelectableListModel
from ert.gui.ertwidgets.storage_info_widget import StorageInfoWidget
from ert.gui.ertwidgets.storage_widget import StorageWidget

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier


class ManageExperimentsPanel(QTabWidget):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier, ensemble_size: int):
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

        parameter_model = SelectableListModel(
            self.ert_config.ensemble_config.parameters
        )
        parameter_check_list = CheckList(parameter_model, "Parameters")
        parameter_check_list.setMinimumWidth(500)
        center_layout.addWidget(parameter_check_list)

        members_model = SelectableListModel(
            [
                str(member)
                for member in range(self.ert_config.model_config.num_realizations)
            ]
        )
        member_check_list = CheckList(members_model, "Members")
        center_layout.addWidget(member_check_list, stretch=1)

        main_layout.addLayout(center_layout)
        main_layout.addSpacing(10)

        initialize_button = QPushButton("Initialize")
        initialize_button.setObjectName("initialize_from_scratch_button")
        initialize_button.setMinimumWidth(75)
        initialize_button.setMaximumWidth(150)

        @showWaitCursorWhileWaiting
        def initialize_from_scratch(_: Any) -> None:
            parameters = parameter_model.getSelectedItems()
            sample_prior(
                ensemble=ensemble_selector.currentData(),
                active_realizations=[int(i) for i in members_model.getSelectedItems()],
                parameters=parameters,
            )

        def update_button_state() -> None:
            initialize_button.setEnabled(ensemble_selector.count() > 0)

        update_button_state()
        ensemble_selector.ensemble_populated.connect(update_button_state)
        initialize_button.clicked.connect(initialize_from_scratch)
        initialize_button.clicked.connect(
            lambda: self._storage_info_widget.setEnsemble(
                ensemble_selector.currentData()
            )
        )
        initialize_button.clicked.connect(ensemble_selector.populate)

        main_layout.addWidget(initialize_button, 1, Qt.AlignmentFlag.AlignRight)
        main_layout.addSpacing(10)
        panel.setLayout(main_layout)

        self.addTab(panel, "Initialize from scratch")

    def eventFilter(self, a0: Optional[QObject], a1: Optional[QEvent]) -> bool:
        if a1 is not None and a1.type() == QEvent.Type.Close:
            self.notifier.emitErtChange()
        return super().eventFilter(a0, a1)
