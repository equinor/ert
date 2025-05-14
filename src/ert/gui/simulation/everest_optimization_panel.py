from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QFormLayout,
    QLabel,
    QWidget,
)

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    CopyableLabel,
    StringBox,
    TextModel,
)
from ert.mode_definitions import ENOPT_MODE, ENSEMBLE_EXPERIMENT_MODE
from ert.validation import ExperimentValidation, ProperNameArgument

from ...run_models.everest_run_model import EverestRunModel
from .experiment_config_panel import ExperimentConfigPanel


@dataclass
class Arguments:
    mode: str
    realizations: str
    current_ensemble: str
    experiment_name: str


class EverestOptimizationPanel(ExperimentConfigPanel):
    def __init__(
        self,
        config_file: str,
        notifier: ErtNotifier,
        run_path: str,
    ):
        super().__init__(EverestRunModel)
        self.notifier = notifier
        self.setObjectName("Everest_optimization_panel")

        layout = QFormLayout()
        lab = QLabel(" ".join(EverestRunModel.__doc__.split()))  # type: ignore
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addRow(lab)

        self._experiment_name_field = StringBox(
            TextModel(""),
            placeholder_text=self.notifier.storage.get_unique_experiment_name(
                ENOPT_MODE
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

        # 2DO adapt for model realizations
        # +------------------------------------------------------------+
        # number_of_realizations_container = QWidget()
        # number_of_realizations_layout = QHBoxLayout(number_of_realizations_container)
        # number_of_realizations_layout.setContentsMargins(0, 0, 0, 0)
        # number_of_realizations_label =
        # QLabel(f"<b>{len(config.model.realizations)}</b>")
        # number_of_realizations_label.setObjectName("num_reals_label")
        # number_of_realizations_layout.addWidget(number_of_realizations_label)

        # layout.addRow(
        #    QLabel("Number of realizations:"), number_of_realizations_container
        # )

        # self._active_realizations_field = StringBox(
        #    ActiveRealizationsModel(ensemble_size),  # type: ignore
        #    "config/simulation/active_realizations",
        # )

        # 2DO deactivate/activate model realizations
        # self._active_realizations_field.setValidator(
        #     RangeStringArgument(len(num_model_realizations)),
        # )
        # layout.addRow("Active realizations", self._active_realizations_field)
        # +------------------------------------------------------------+

        # No need for design matrix(?)

        self.setLayout(layout)

        # self._active_realizations_field.getValidationSupport()
        # .validationChanged.connect(
        #     self.simulationConfigurationChanged
        # )
        self._experiment_name_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )

        # 2DO remove hardcoding of batch_
        # self._ensemble_name_field.getValidationSupport().validationChanged.connect(
        #    self.simulationConfigurationChanged
        # )

        self.notifier.ertChanged.connect(self._update_experiment_name_placeholder)

    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> None:
        if isinstance(w, EverestOptimizationPanel):
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
            # self._active_realizations_field.isValid()
            # and
            self._experiment_name_field.isValid()
            # and self._ensemble_name_field.isValid()
        )

    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ENOPT_MODE,
            # 2DO fixup some hardcoding of batch_N
            current_ensemble=None,  # self._ensemble_name_field.get_text,
            realizations=None,  # self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
        )
