from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from qtpy import QtCore
from qtpy.QtCore import QEvent, QModelIndex, QObject, Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from typing_extensions import override

from ert.enkf_main import sample_prior
from ert.gui.ertwidgets import (
    CheckList,
    EnsembleSelector,
    SelectableListModel,
    showWaitCursorWhileWaiting,
)
from ert.sensitivity_analysis.design_matrix import (
    initialize_parameters,
    read_design_matrix,
)

from ...ertwidgets.create_experiment_dialog import CreateExperimentDialog
from .storage_info_widget import StorageInfoWidget
from .storage_widget import StorageWidget

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier


class DFModel(QtCore.QAbstractTableModel):
    def __init__(self, data: pd.DataFrame, parent: Optional[QWidget] = None) -> None:
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._df = data

    @override
    def data(
        self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> QtCore.QVariant:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return QtCore.QVariant(str(self._df.iloc[index.row()].iloc[index.column()]))
        return QtCore.QVariant()

    @override
    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return len(self._df.values)

    @override
    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        return self._df.columns.size

    @override
    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Optional[int] = None
    ) -> QtCore.QVariant | str:
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return "\n".join(self._df.columns[section])
        return QtCore.QVariant()


def createRow(*widgets: CheckList) -> QHBoxLayout:
    row = QHBoxLayout()

    for widget in widgets:
        row.addWidget(widget)

    row.addStretch()
    return row


class ManageExperimentsPanel(QTabWidget):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier, ensemble_size: int):
        QTabWidget.__init__(self)
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self.notifier = notifier

        self._add_create_new_ensemble_tab()
        self._add_initialize_from_scratch_tab()

        self.installEventFilter(self)
        if self.ert_config.analysis_config.design_matrix is not None:
            self._add_initialize_from_design_matrix_tab()

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
                random_seed=self.ert_config.random_seed,
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

    def _add_initialize_from_design_matrix_tab(self) -> None:
        assert self.ert_config.analysis_config.design_matrix is not None  # for mypy
        try:
            design_matrix = read_design_matrix(
                self.ert_config,
                self.ert_config.analysis_config.design_matrix,
            )
        except (ValueError, KeyError, OSError) as err:
            QMessageBox.warning(
                self,
                "Warning",
                (
                    "The following issue where found when attempting to read the "
                    f'design matrix: "{err}"'
                ),
            )
            return
        panel = QWidget()
        panel.setObjectName("initialize_from_design_matrix_panel")
        layout = QVBoxLayout()
        view = QTableView()
        self.pandas_model = DFModel(design_matrix)
        view.setModel(self.pandas_model)
        layout.addWidget(view)

        initialize_button = QPushButton("Initialize")
        initialize_button.setObjectName("initialize_from_design_button")
        initialize_button.setMinimumWidth(75)
        initialize_button.setMaximumWidth(150)

        @showWaitCursorWhileWaiting
        def initializeFromDesignMatrix(_: Any) -> None:
            create_experiment_dialog = CreateExperimentDialog(
                parent=self, notifier=self.notifier
            )
            create_experiment_dialog.show()
            if create_experiment_dialog.exec_():
                try:
                    ensemble = initialize_parameters(
                        self.pandas_model._df,
                        self.notifier.storage,
                        self.ert_config,
                        exp_name=create_experiment_dialog.experiment_name,
                        ens_name=create_experiment_dialog.ensemble_name,
                    )
                    self.notifier.set_current_ensemble(ensemble)
                    self.notifier.ertChanged.emit()
                except (ValueError, KeyError, OSError) as err:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        (
                            "Something went wrong when initializing the ensemble "
                            f'got "{err}"'
                        ),
                    )

        initialize_button.clicked.connect(initializeFromDesignMatrix)

        layout.addWidget(initialize_button, 0, Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(10)

        panel.setLayout(layout)
        self.addTab(panel, "Initialize from design matrix")
