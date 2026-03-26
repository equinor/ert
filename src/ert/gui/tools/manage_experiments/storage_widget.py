import logging
import re
from collections.abc import Callable
from typing import cast

from PyQt6.QtCore import (
    QAbstractItemModel,
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
)
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErrorInfo, ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import CreateExperimentDialog, Suggestor
from ert.storage import Ensemble, Experiment
from ert.storage.local_experiment import ExperimentType

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

        self.addButton = QPushButton("Add new experiment", self)
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
        left_display = left.data(Qt.ItemDataRole.DisplayRole)
        right_display = right.data(Qt.ItemDataRole.DisplayRole)
        if (
            isinstance(left_display, str)
            and left_display.startswith("Realization ")
            and isinstance(right_display, str)
            and right_display.startswith("Realization ")
        ):
            try:
                left_num = int(left_display.split(" ")[1])
                right_num = int(right_display.split(" ")[1])
            except (IndexError, ValueError):
                return left_display.lower() < right_display.lower()
            return left_num < right_num

        role = int(self.sortRole())
        left_data = left.data(role)
        right_data = right.data(role)

        if left_data is None and right_data is None:
            return False
        if left_data is None:
            return False
        if right_data is None:
            return True

        if isinstance(left_data, str) and isinstance(right_data, str):

            def natural_key(s: str) -> list[int | str]:
                return [
                    int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)
                ]

            lk = natural_key(left_data)
            rk = natural_key(right_data)
            if lk == rk:
                return left_data.lower() < right_data.lower()
            return lk < rk

        return left_data < right_data


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
        proxy_model.setSortRole(Qt.ItemDataRole.UserRole)
        proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        proxy_model.setDynamicSortFilter(True)
        proxy_model.sort(1, Qt.SortOrder.DescendingOrder)

        self._tree_view.setModel(proxy_model)
        self._tree_view.setSortingEnabled(True)
        header = self._tree_view.header()
        if header is not None:
            header.setSortIndicatorShown(True)
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

        if self._notifier.is_experiment_running:
            disableAdd()

        notifier.experiment_started.connect(disableAdd)
        notifier.experiment_ended.connect(enableAdd)

        layout = QVBoxLayout()
        layout.addWidget(search_bar)
        layout.addWidget(self._tree_view)
        layout.addWidget(self._create_experiment_button)

        self.setLayout(layout)

    def _currentChanged(self, selected: QModelIndex, previous: QModelIndex) -> None:
        model = cast(_SortingProxyModel, self._tree_view.model())
        idx = model.mapToSource(selected)
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
            try:
                with self._notifier.write_storage() as storage:
                    parameter_configuration = (
                        self._ert_config.parameter_configurations_with_design_matrix
                    )
                    response_configuration = (
                        self._ert_config.ensemble_config.response_configuration
                    )
                    ensemble = storage.create_experiment(
                        experiment_config={
                            "parameter_configuration": [
                                c.model_dump(mode="json")
                                for c in parameter_configuration
                            ],
                            "response_configuration": [
                                c.model_dump(mode="json")
                                for c in response_configuration
                            ],
                            "observations": [
                                d.model_dump(mode="json")
                                for d in self._ert_config.observation_declarations
                            ],
                            "ert_templates": self._ert_config.ert_templates,
                            "experiment_type": ExperimentType.MANUAL,
                            "shape_registry": self._ert_config.shape_registry,
                        },
                        name=create_experiment_dialog.experiment_name,
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
