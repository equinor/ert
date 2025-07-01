import logging
from collections.abc import Callable

from PyQt6.QtCore import (
    QAbstractItemModel,
    QItemSelectionModel,
    QModelIndex,
    QSize,
    QSortFilterProxyModel,
    Qt,
)
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ert.config import ConfigValidationError, ErrorInfo, ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.create_experiment_dialog import CreateExperimentDialog
from ert.gui.suggestor import Suggestor
from ert.storage import Ensemble, Experiment

from .storage_model import (
    EnsembleModel,
    ExperimentModel,
    RealizationModel,
    StorageModel,
)

logger = logging.getLogger(__name__)


class AddWidget(QWidget):
    """
    A widget with an add button.
    Parameters
    ----------
    addFunction: Callable to be connected to the add button.
    """

    def __init__(self, addFunction: Callable[[], None]) -> None:
        super().__init__()

        self.addButton = QToolButton(self)
        self.addButton.setIcon(QIcon("img:add_circle_outlined.svg"))
        self.addButton.setIconSize(QSize(16, 16))
        self.addButton.clicked.connect(addFunction)

        self.removeButton = None

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.addStretch(1)
        self.buttonLayout.addWidget(self.addButton)
        self.buttonLayout.addSpacing(2)

        self.setLayout(self.buttonLayout)


class _SortingProxyModel(QSortFilterProxyModel):
    def __init__(self, model: QAbstractItemModel) -> None:
        super().__init__()
        self.setSourceModel(model)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        left_data = left.data()
        right_data = right.data()

        if (
            isinstance(left_data, str)
            and "Realization" in left_data
            and isinstance(right_data, str)
            and "Realization" in right_data
        ):
            left_realization_number = int(left_data.split(" ")[1])
            right_realization_number = int(right_data.split(" ")[1])

            return left_realization_number < right_realization_number

        return super().lessThan(left, right)


class StorageWidget(QWidget):
    onSelectEnsemble = Signal(Ensemble)
    onSelectExperiment = Signal(Experiment)
    onSelectRealization = Signal(Ensemble, int)

    def __init__(
        self, notifier: ErtNotifier, ert_config: ErtConfig, ensemble_size: int
    ) -> None:
        QWidget.__init__(self)

        self._notifier = notifier
        self._ert_config = ert_config
        self._ensemble_size = ensemble_size
        self.setMinimumWidth(500)

        self._tree_view = QTreeView(self)
        storage_model = StorageModel(self._notifier.storage)
        notifier.ertChanged.connect(
            lambda: storage_model.reloadStorage(self._notifier.storage)
        )

        search_bar = QLineEdit(self)
        search_bar.setPlaceholderText("Filter")
        proxy_model = _SortingProxyModel(storage_model)
        proxy_model.setFilterKeyColumn(-1)  # Search all columns.
        proxy_model.setSourceModel(storage_model)
        proxy_model.sort(0, Qt.SortOrder.AscendingOrder)

        self._tree_view.setModel(proxy_model)
        search_bar.textChanged.connect(proxy_model.setFilterFixedString)

        self._sel_model = QItemSelectionModel(proxy_model)
        self._sel_model.currentChanged.connect(self._currentChanged)
        self._tree_view.setSelectionModel(self._sel_model)
        self._tree_view.setColumnWidth(0, 225)
        self._tree_view.setColumnWidth(1, 125)
        self._tree_view.setColumnWidth(2, 100)

        self._create_experiment_button = AddWidget(self._addItem)

        @Slot()
        def disableAdd() -> None:
            self._create_experiment_button.setEnabled(False)

        @Slot()
        def enableAdd() -> None:
            self._create_experiment_button.setEnabled(True)

        if self._notifier.is_simulation_running:
            disableAdd()

        notifier.simulationStarted.connect(disableAdd)
        notifier.simulationEnded.connect(enableAdd)

        layout = QVBoxLayout()
        layout.addWidget(search_bar)
        layout.addWidget(self._tree_view)
        layout.addWidget(self._create_experiment_button)

        self.setLayout(layout)

    def _currentChanged(self, selected: QModelIndex, previous: QModelIndex) -> None:
        idx = self._tree_view.model().mapToSource(selected)  # type: ignore
        cls = idx.internalPointer()

        if isinstance(cls, EnsembleModel):
            ensemble = self._notifier.storage.get_ensemble(cls._id)
            self.onSelectEnsemble.emit(ensemble)
        elif isinstance(cls, ExperimentModel):
            experiment = self._notifier.storage.get_experiment(cls._id)
            self.onSelectExperiment.emit(experiment)
        elif isinstance(cls, RealizationModel):
            ensemble = self._notifier.storage.get_ensemble(cls.ensemble_id)
            self.onSelectRealization.emit(ensemble, cls.realization)

    def _addItem(self) -> None:
        create_experiment_dialog = CreateExperimentDialog(self._notifier, parent=self)
        create_experiment_dialog.show()
        if create_experiment_dialog.exec():
            parameters_config = self._ert_config.ensemble_config.parameter_configuration
            design_matrix = self._ert_config.analysis_config.design_matrix
            design_matrix_group = None
            if design_matrix is not None:
                try:
                    parameters_config, design_matrix_group = (
                        design_matrix.merge_with_existing_parameters(parameters_config)
                    )
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
            try:
                with self._notifier.write_storage() as storage:
                    ensemble = storage.create_experiment(
                        parameters=(
                            [*parameters_config, design_matrix_group]
                            if design_matrix_group is not None
                            else parameters_config
                        ),
                        responses=self._ert_config.ensemble_config.response_configuration,
                        observations=self._ert_config.enkf_obs.datasets,
                        name=create_experiment_dialog.experiment_name,
                        templates=self._ert_config.ert_templates,
                    ).create_ensemble(
                        name=create_experiment_dialog.ensemble_name,
                        ensemble_size=self._ensemble_size,
                        iteration=create_experiment_dialog.iteration,
                    )

                self._notifier.set_current_ensemble_id(ensemble.id)
            except OSError as err:
                logger.error(str(err))
                Suggestor(
                    errors=[ErrorInfo(str(err))],
                    widget_info=(
                        '<p style="font-size: 28px;">'
                        "Error writing to storage, experiment not created</p>"
                    ),
                    parent=self,
                ).show()
