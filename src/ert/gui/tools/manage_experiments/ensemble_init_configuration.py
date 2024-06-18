from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from PyQt5.QtWidgets import QTableView
from qtpy import QtCore
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import (
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
from ert.gui.ertwidgets.create_experiment_dialog import CreateExperimentDialog
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.ertwidgets.models.selectable_list_model import SelectableListModel
from ert.gui.ertwidgets.storage_info_widget import StorageInfoWidget
from ert.gui.ertwidgets.storage_widget import StorageWidget
from ert.gui.tools.manage_experiments.design_matrix import (
    initialize_parameters,
    read_design_matrix,
)

if TYPE_CHECKING:
    from ert.gui.ertnotifier import ErtNotifier


class DFModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._df = data

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return QtCore.QVariant(str(self._df.iloc[index.row()][index.column()]))
        return QtCore.QVariant()

    def rowCount(self, parent=None):
        return len(self._df.values)

    def columnCount(self, parent=None):
        return self._df.columns.size

    def headerData(self, col, orientation, role=None):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return "\n".join(self._df.columns[col])


def createCheckLists(
    ensemble_size: int, parameters: List[str]
) -> Tuple[QHBoxLayout, SelectableListModel, SelectableListModel]:
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


def createRow(*widgets: CheckList) -> QHBoxLayout:
    row = QHBoxLayout()

    for widget in widgets:
        row.addWidget(widget)

    row.addStretch()
    return row


class EnsembleInitializationConfigurationPanel(QTabWidget):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier, ensemble_size: int):
        QTabWidget.__init__(self)
        self.ert_config = config
        self.ensemble_size = ensemble_size
        self.notifier = notifier
        self._addCreateNewEnsembleTab()
        self._addInitializeFromScratchTab()
        self.installEventFilter(self)
        if self.ert_config.analysis_config.design_matrix:
            self._addInitializeFromDesignMatrixTab()

        self.setWindowTitle("Manage experiments")
        self.setMinimumWidth(850)
        self.setMinimumHeight(250)

    def _addCreateNewEnsembleTab(self) -> None:
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

    def _addInitializeFromScratchTab(self) -> None:
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
        def initializeFromScratch(_) -> None:
            parameters = parameter_model.getSelectedItems()
            sample_prior(
                ensemble=target_ensemble.currentData(),
                active_realizations=[int(i) for i in members_model.getSelectedItems()],
                parameters=parameters,
            )

        def update_button_state() -> None:
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

    def eventFilter(self, watched: QObject, event: QEvent):
        if event.type() == QEvent.Type.Close:
            self.notifier.emitErtChange()
        return super().eventFilter(watched, event)

    def _addInitializeFromDesignMatrixTab(self):
        panel = QWidget()
        panel.setObjectName("initialize_from_design_matrix_panel")
        layout = QVBoxLayout()

        design_matrix = read_design_matrix(
            self.ert_config,
            self.ert_config.analysis_config.design_matrix,
        )
        view = QTableView()
        self.pandas_model = DFModel(design_matrix)
        view.setModel(self.pandas_model)
        layout.addWidget(view)

        initialize_button = QPushButton("Initialize")
        initialize_button.setObjectName("initialize_from_design_button")
        initialize_button.setMinimumWidth(75)
        initialize_button.setMaximumWidth(150)

        @showWaitCursorWhileWaiting
        def initializeFromDesignMatrix(_):
            create_experiment_dialog = CreateExperimentDialog(parent=self)
            create_experiment_dialog.show()
            if create_experiment_dialog.exec_():
                ensemble = initialize_parameters(
                    self.pandas_model._df,
                    self.notifier.storage,
                    self.ert_config,
                    exp_name=create_experiment_dialog.experiment_name,
                    ens_name=create_experiment_dialog.ensemble_name,
                )
                self.notifier.set_current_ensemble(ensemble)
                self.notifier.ertChanged.emit()

        initialize_button.clicked.connect(initializeFromDesignMatrix)

        layout.addWidget(initialize_button, 0, Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(10)

        panel.setLayout(layout)
        self.addTab(panel, "Initialize from design matrix")
