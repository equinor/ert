from dataclasses import dataclass

from qtpy import QtCore
from qtpy.QtCore import Slot
from qtpy.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QPushButton

from ert.config import AnalysisConfig, DesignMatrix
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    CopyableLabel,
    StringBox,
    TextModel,
)
from ert.gui.tools.design_matrix.design_matrix_panel import DesignMatrixPanel
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from ert.run_models import EnsembleExperiment
from ert.validation import RangeStringArgument
from ert.validation.proper_name_argument import ExperimentValidation, ProperNameArgument

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
        ensemble_size: int,
        run_path: str,
        notifier: ErtNotifier,
    ):
        self.notifier = notifier
        super().__init__(EnsembleExperiment)
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()
        lab = QLabel(" ".join(EnsembleExperiment.__doc__.split()))  # type: ignore
        lab.setWordWrap(True)
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
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

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(ensemble_size),  # type: ignore
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(ensemble_size),
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        design_matrix = analysis_config.design_matrix
        if design_matrix is not None:
            show_dm_param_button = QPushButton("Show parameters")
            show_dm_param_button.setObjectName("show-dm-parameters")
            show_dm_param_button.setMinimumWidth(50)

            button_layout = QHBoxLayout()
            button_layout.addWidget(show_dm_param_button)
            button_layout.addStretch()  # Add stretch to push the button to the left

            layout.addRow("Design Matrix", button_layout)
            show_dm_param_button.clicked.connect(
                lambda: self.on_show_dm_params_clicked(design_matrix)
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

    def on_show_dm_params_clicked(self, design_matrix: DesignMatrix) -> None:
        assert design_matrix is not None

        if design_matrix.design_matrix_df is None:
            design_matrix.read_design_matrix()

        if (
            design_matrix.design_matrix_df is not None
            and not design_matrix.design_matrix_df.empty
        ):
            viewer = DesignMatrixPanel(
                design_matrix.design_matrix_df,
                design_matrix.xls_filename.name,
            )
            viewer.setMinimumHeight(500)
            viewer.setMinimumWidth(1000)
            viewer.adjustSize()
            viewer.exec_()

    @Slot(ExperimentConfigPanel)
    def experimentTypeChanged(self, w: ExperimentConfigPanel) -> None:
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
