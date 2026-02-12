import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget
from typing_extensions import override

from ert.config import AnalysisConfig, ErrorInfo
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    AnalysisModuleEdit,
    CopyableLabel,
    EnsembleSelector,
    StringBox,
    Suggestor,
    TargetEnsembleModel,
)
from ert.gui.simulation.experiment_config_panel import ExperimentConfigPanel
from ert.mode_definitions import MANUAL_ENIF_UPDATE_MODE, MANUAL_UPDATE_MODE
from ert.run_models.manual_update import ManualUpdate
from ert.storage import Ensemble, RealizationStorageState
from ert.validation import EnsembleRealizationsArgument, ProperNameFormatArgument

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    mode: str
    realizations: str
    ensemble_id: str
    target_ensemble: str
    ensemble_size: int


class ManualUpdatePanel(ExperimentConfigPanel):
    updateMethodChanged = pyqtSignal(str)

    def __init__(
        self,
        run_path: str,
        notifier: ErtNotifier,
        analysis_config: AnalysisConfig,
    ) -> None:
        self.notifier = notifier
        super().__init__(ManualUpdate)
        self.setObjectName("Manual_update_panel")

        layout = QFormLayout()
        self._update_method_dropdown = QComboBox()
        self._update_method_dropdown.addItems(
            ["ES Update", "EnIF Update (Experimental)"]
        )
        self._update_method_dropdown.currentTextChanged.connect(
            self._on_update_method_changed
        )
        self._update_method_dropdown.setObjectName("manual_update_method_dropdown")

        layout.addRow("Update method:", self._update_method_dropdown)

        self._ensemble_selector = EnsembleSelector(
            notifier, show_only_with_response_data=True
        )
        layout.addRow("Ensemble:", self._ensemble_selector)
        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        self._number_of_realizations_label = QLabel()
        layout.addRow(
            QLabel("Number of realizations:"),
            self._number_of_realizations_label,
        )

        self._ensemble_format_model = TargetEnsembleModel(analysis_config, notifier)
        self._ensemble_format_field = StringBox(
            self._ensemble_format_model,  # type: ignore
            self._ensemble_format_model.getDefaultValue(),  # type: ignore
            continuous_update=True,
        )
        self._ensemble_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Ensemble format:", self._ensemble_format_field)

        self._analysis_module_edit = AnalysisModuleEdit(analysis_config.es_settings, 0)
        self._analysis_module_edit.setObjectName("ensemble_smoother_edit")
        self._analysis_module_edit.setEnabled(False)
        layout.addRow("Analysis module:", self._analysis_module_edit)
        self._active_realizations_model = ActiveRealizationsModel(0, show_default=False)
        self._active_realizations_field = StringBox(
            self._active_realizations_model,  # type: ignore
            continuous_update=True,
        )
        self._active_realizations_field.setObjectName("active_realizations_box")
        self._realizations_from_fs()
        layout.addRow("Active realizations", self._active_realizations_field)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._ensemble_selector.ensemble_populated.connect(self._realizations_from_fs)
        self._ensemble_selector.ensemble_populated.connect(
            self.simulationConfigurationChanged
        )
        self._ensemble_selector.currentIndexChanged.connect(self._realizations_from_fs)
        self.setLayout(layout)

    @property
    def selected_update_method(self) -> str:
        return self._update_method_dropdown.currentText()

    @Slot(str)
    def _on_update_method_changed(self, new_method: str) -> None:
        if new_method == "ES Update":
            self._analysis_module_edit.show()
        else:
            self._analysis_module_edit.hide()

    @override
    def isConfigurationValid(self) -> bool:
        return (
            self._active_realizations_field.isValid()
            and self._ensemble_selector.currentIndex() != -1
        )

    @override
    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=MANUAL_UPDATE_MODE
            if self.selected_update_method == "ES Update"
            else MANUAL_ENIF_UPDATE_MODE,
            ensemble_id=str(
                cast(Ensemble, self._ensemble_selector.selected_ensemble).id
            ),
            realizations=self._active_realizations_field.text(),
            target_ensemble=self._ensemble_format_model.getValue(),  # type: ignore
            ensemble_size=self._ensemble_size,
        )

    def _realizations_from_fs(self) -> None:
        ensemble = self._ensemble_selector.selected_ensemble
        self._active_realizations_field.setEnabled(ensemble is not None)
        try:
            if ensemble:
                parameters = ensemble.get_realization_mask_with_parameters()
                responses = ensemble.get_realization_mask_with_responses()
                mask = np.logical_and(parameters, responses)
                self._ensemble_size = ensemble.ensemble_size
                self._active_realizations_field.setValidator(
                    EnsembleRealizationsArgument(
                        lambda: ensemble,
                        required_realization_storage_states=[
                            RealizationStorageState.PARAMETERS_LOADED,
                            RealizationStorageState.RESPONSES_LOADED,
                        ],
                    )
                )
                self._active_realizations_model.ensemble_size = ensemble.ensemble_size
                self._active_realizations_model.setValueFromMask(mask)
                self._number_of_realizations_label.setText(
                    f"<b>{ensemble.ensemble_size}</b>"
                )
                # only use active realizations for setting threshold
                active_realizations_size = sum(
                    self._active_realizations_model.getActiveRealizationsMask()
                )
                self._analysis_module_edit.ensemble_size = active_realizations_size
                self._analysis_module_edit.setEnabled(bool(active_realizations_size))
        except OSError as err:
            logger.error(str(err))
            Suggestor(
                errors=[ErrorInfo(str(err))],
                widget_info='<p style="font-size: 28px;">Error reading storage</p>',
                parent=self,
            ).show()

    @override
    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> None:
        if isinstance(w, ManualUpdatePanel):
            self._realizations_from_fs()
