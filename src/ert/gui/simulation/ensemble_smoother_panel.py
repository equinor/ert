from dataclasses import dataclass

from qtpy.QtWidgets import QFormLayout, QLabel

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import AnalysisModuleEdit
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.targetcasemodel import TargetCaseModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.libres_facade import LibresFacade
from ert.run_models import EnsembleSmoother
from ert.validation import ProperNameFormatArgument, RangeStringArgument

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    target_case: str
    realizations: str
    current_case: str


class EnsembleSmootherPanel(SimulationConfigPanel):
    def __init__(
        self, facade: LibresFacade, notifier: ErtNotifier, ensemble_size: int
    ) -> None:
        super().__init__(EnsembleSmoother)
        self.notifier = notifier
        layout = QFormLayout()

        self.setObjectName("ensemble_smoother_panel")

        runpath_label = CopyableLabel(text=facade.run_path_stripped)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._case_format_model = TargetCaseModel(facade, notifier, format_mode=True)
        self._case_format_field = StringBox(
            self._case_format_model,
            self._case_format_model.getDefaultValue(),
            True,
        )
        self._case_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Case format:", self._case_format_field)

        self._analysis_module_edit = AnalysisModuleEdit(
            facade.get_analysis_module("STD_ENKF"), ensemble_size
        )
        self._analysis_module_edit.setObjectName("ensemble_smoother_edit")
        layout.addRow("Analysis module:", self._analysis_module_edit)

        active_realizations_model = ActiveRealizationsModel(ensemble_size)
        self._active_realizations_field = StringBox(
            active_realizations_model, "config/simulation/active_realizations"
        )
        self._active_realizations_field.setValidator(RangeStringArgument(ensemble_size))
        layout.addRow("Active realizations", self._active_realizations_field)

        self.setLayout(layout)

        self._case_format_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )

    def isConfigurationValid(self) -> bool:
        return (
            self._case_format_field.isValid()
            and self._active_realizations_field.isValid()
        )

    def getSimulationArguments(self) -> Arguments:
        arguments = Arguments(
            mode="ensemble_smoother",
            current_case=self._case_format_model.getValue() % 0,
            target_case=self._case_format_model.getValue() % 1,
            realizations=self._active_realizations_field.text(),
        )
        return arguments
