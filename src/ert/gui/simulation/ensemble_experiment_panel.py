from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from ert.config import AnalysisConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    CopyableLabel,
    StringBox,
    TextModel,
)
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from ert.run_models import EnsembleExperiment
from ert.validation import ExperimentValidation, ProperNameArgument
from ert.validation.active_range import ActiveRange
from ert.validation.range_string_argument import RangeSubsetStringArgument

from ._design_matrix_panel import DesignMatrixPanel
from .experiment_config_panel import ExperimentConfigPanel


@dataclass
class Arguments:
    mode: str
    realizations: str
    current_ensemble: str
    experiment_name: str


class EnsembleExperimentPanel(ExperimentConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        active_realizations: list[bool],
        config_num_realization: int,
        run_path: str,
        notifier: ErtNotifier,
    ) -> None:
        super().__init__(EnsembleExperiment)
        self.notifier = notifier
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()
        lab = QLabel(" ".join(EnsembleExperiment.__doc__.split()))  # type: ignore
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addRow(lab)

        self._experiment_name_field = StringBox(
            TextModel(""),
            placeholder_text=self.notifier.storage.get_unique_experiment_name(
                ENSEMBLE_EXPERIMENT_MODE
            ),
        )
        self._experiment_name_field.setMinimumWidth(250)
        self._experiment_name_field.setValidator(
            ExperimentValidation(self.notifier.storage)
        )
        self._experiment_name_field.setObjectName("experiment_field")
        layout.addRow("Experiment name:", self._experiment_name_field)

        self._ensemble_name_field = StringBox(
            TextModel(""), placeholder_text="ensemble"
        )
        self._ensemble_name_field.setValidator(ProperNameArgument())
        self._ensemble_name_field.setMinimumWidth(250)

        layout.addRow("Ensemble name:", self._ensemble_name_field)

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_container = QWidget()
        number_of_realizations_layout = QHBoxLayout(number_of_realizations_container)
        number_of_realizations_layout.setContentsMargins(0, 0, 0, 0)
        number_of_realizations_label = QLabel(f"<b>{len(active_realizations)}</b>")
        number_of_realizations_label.setObjectName("num_reals_label")
        number_of_realizations_layout.addWidget(number_of_realizations_label)

        layout.addRow(
            QLabel("Number of realizations:"), number_of_realizations_container
        )

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(len(active_realizations)),  # type: ignore
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(
            RangeSubsetStringArgument(ActiveRange(active_realizations)),
        )
        self._active_realizations_field.model.setValueFromMask(  # type: ignore
            active_realizations
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        design_matrix = analysis_config.design_matrix
        if design_matrix is not None:
            layout.addRow(
                "Design Matrix",
                DesignMatrixPanel.get_design_matrix_button(
                    design_matrix,
                    number_of_realizations_label,
                    config_num_realization,
                ),
            )

        self.setLayout(layout)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._experiment_name_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._ensemble_name_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )

        self.notifier.ertChanged.connect(self._update_experiment_name_placeholder)

    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> None:
        if isinstance(w, EnsembleExperimentPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(ENSEMBLE_EXPERIMENT_MODE)
        )

    def isConfigurationValid(self) -> bool:
        self.blockSignals(True)
        self._experiment_name_field.validateString()
        self._ensemble_name_field.validateString()
        self.blockSignals(False)
        return (
            self._active_realizations_field.isValid()
            and self._experiment_name_field.isValid()
            and self._ensemble_name_field.isValid()
        )

    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ENSEMBLE_EXPERIMENT_MODE,
            current_ensemble=self._ensemble_name_field.get_text,
            realizations=self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
        )
